"""
NEURON.PY - Electrophysiology Data Acquisition and Analysis Platform
=====================================================================

Python reverse-engineered version of Neuron.m (MATLAB)
Complete neuroscience data acquisition, analysis, and visualization system

Original: ~30,000 lines MATLAB with 515 functions
Python Version: Object-oriented architecture with modern libraries

Author: Reverse engineered from Neuron.m
Version: 1.020 (Python port)
"""

import sys
import os
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import h5py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import json
import time
from datetime import datetime
from enum import Enum
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QFileDialog, QMessageBox, QProgressBar, QTextEdit,
                             QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QSlider, QTabWidget, QGroupBox, QDialog, QDialogButtonBox,
                             QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QPalette

# Import EEGLoader from the same directory
try:
    from eeg_loader import EEGLoader
except ImportError:
    print("Warning: eeg_loader.py not found. .dat file loading will not work.")
    EEGLoader = None


# ============================================================================
# CONSTANTS AND ENUMERATIONS
# ============================================================================

class AcquisitionMode(Enum):
    """Data acquisition modes"""
    OFFLINE = 0
    SWP = 1      # Sweep mode
    EEG = 2      # Continuous recording
    TST = 3      # Test mode
    MEA = 4      # Multi-electrode array


class ButtonState(Enum):
    """Button/knob color states"""
    GREY = 'grey'
    WHITE = 'white'
    DARK = 'dark'
    RED = 'red'
    BLUE = 'blue'
    GREEN = 'green'
    YELLOW = 'yellow'


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class NeuronColors:
    """Color definitions matching MATLAB structure"""
    zwart: Tuple[float, float, float] = (0, 0, 0)
    wit: Tuple[float, float, float] = (1, 1, 1)
    blauw: Tuple[float, float, float] = (0, 0, 1)
    groen: Tuple[float, float, float] = (0, 1, 0)
    rood: Tuple[float, float, float] = (1, 0, 0)
    geel: Tuple[float, float, float] = (1, 1, 0)
    grey: Tuple[float, float, float] = (0.85, 0.85, 0.85)
    dark: Tuple[float, float, float] = (0.75, 0.75, 0.75)
    red: Tuple[float, float, float] = (1, 0.8, 0.8)
    blue: Tuple[float, float, float] = (0.5, 1, 1)
    yellow: Tuple[float, float, float] = (0.5, 0.5, 0)
    green: Tuple[float, float, float] = (0.5, 1, 0.5)


@dataclass
class Zoom:
    """Zoom parameters"""
    pnts: int = 0
    rate: float = 1.0
    tnul: float = 0.0
    tend: float = 0.0
    tbgn: float = 0.0
    tlst: float = 0.0


@dataclass
class Cursor:
    """Cursor measurement tool data"""
    absnul: float = 0.0
    handle3: Any = None
    cross: Any = None
    crossset: Any = None
    crosslines: Any = None
    times: Tuple[float, float] = (0, 0)
    index: Tuple[int, int] = (0, 0)
    rate: float = 0.0
    text1: Any = None
    text2: Any = None
    text3: Any = None
    channel: int = 1
    cpy: float = 0.0
    hhh: Any = None
    femke: Any = None
    fline: Any = None
    smooth: bool = True


@dataclass
class VideoSettings:
    """Video synchronization settings"""
    cage: int = 0
    name: Optional[str] = None
    time: float = 0.0
    segnr: int = 0
    zoom: Any = None
    VLCpos: Any = None


@dataclass
class FileSettings:
    """File I/O settings"""
    name: str = 'C:\\NRNdata\\*.mat'
    scale: float = 1000.0
    unit: str = 'mV'
    txtmax: int = 900
    savepath: Optional[str] = None
    dir: Optional[str] = None
    segdir: Optional[str] = None
    video: Optional[str] = None
    root: str = 'C:\\NRNdata\\'
    script: str = './scripts/'
    proto: str = './protocols/'
    mcc: str = './protocols/MCC/'


@dataclass
class DAQDevice:
    """Data acquisition device configuration"""
    type: str = ''
    BoardName: str = ''
    InstalledBoardId: int = 0
    adc: List[int] = field(default_factory=list)
    dac: List[int] = field(default_factory=list)
    DIO: Dict = field(default_factory=dict)
    TRG: Dict = field(default_factory=dict)
    MCC: Dict = field(default_factory=dict)
    ADC: List[Dict] = field(default_factory=list)
    DAC: List[Dict] = field(default_factory=list)


@dataclass
class ADCChannel:
    """ADC channel configuration"""
    device: int = 0
    local: int = 0
    hwchan: int = 0
    gain: float = 1.0
    usermax: float = 100.0
    name: str = 'adc'
    unit: str = 'mV'
    junction: float = 0.0
    color: str = 'k'


@dataclass
class DACChannel:
    """DAC channel configuration"""
    device: int = 0
    local: int = 0
    hwchan: int = 0
    usermax: float = 100.0
    name: str = 'dac'
    unit: str = 'mV'
    junction: float = 0.0
    color: str = 'k'


@dataclass
class Meting:
    """Measurement data structure (primary data container)"""
    adc: np.ndarray = field(default_factory=lambda: np.array([]))
    dac: np.ndarray = field(default_factory=lambda: np.array([]))
    ADC: List[ADCChannel] = field(default_factory=list)
    DAC: List[DACChannel] = field(default_factory=list)
    setnr: int = 0
    serienaam: str = ''
    nodenaam: str = ''
    plotter: Dict = field(default_factory=dict)
    result: Dict = field(default_factory=dict)
    EXTRA: Dict = field(default_factory=dict)
    time: float = 0.0
    datetime: str = ''
    protocol: str = ''


@dataclass
class SpikeData:
    """Spike detection and template matching data"""
    err: Optional[np.ndarray] = None
    gain: Optional[float] = None
    off: Optional[float] = None
    fact: Optional[float] = None
    template: Optional[np.ndarray] = None
    tmpnul: Optional[int] = None
    tmprng: Optional[Tuple[int, int]] = None
    adcchan: int = 0
    adcrate: float = 0.0
    adcdrmp: float = 0.0
    adcunit: str = ''


@dataclass
class EventData:
    """Event database entry"""
    dbase: List = field(default_factory=list)
    segdir1: Optional[str] = None
    segdir2: Optional[str] = None
    files: Dict = field(default_factory=dict)
    tmp: Dict = field(default_factory=dict)


@dataclass
class AxonAmplifier:
    """Axon amplifier configuration"""
    Vh1: float = np.nan
    Ih1: float = np.nan
    Vs1: float = np.nan
    Vr1: float = np.nan
    gm1: float = np.nan
    Vh2: float = np.nan
    Ih2: float = np.nan
    Vs2: float = np.nan
    Vr2: float = np.nan
    gm2: float = np.nan
    Int1: float = np.nan
    Int2: float = np.nan
    input: List = field(default_factory=list)
    output: List = field(default_factory=list)
    stim: List = field(default_factory=list)
    monitor: int = 0
    interval: float = 0.1
    disprate: float = 0.5
    show: int = 0


@dataclass
class Timeline:
    """Timeline markers and events"""
    red: List = field(default_factory=list)
    blue: List = field(default_factory=list)
    green: List = field(default_factory=list)
    boundery: Tuple[float, float] = (-np.inf, np.inf)
    left: bool = True
    kxy: Any = None
    kpar: Any = None
    kidx: Any = None
    par: Any = None
    idx: Any = None
    xy: Any = None


@dataclass
class MEAData:
    """Multi-electrode array data"""
    data: Optional[np.ndarray] = None
    electrodes: List = field(default_factory=list)
    sampling_rate: float = 0.0
    stim_times: List = field(default_factory=list)


# ============================================================================
# NEURON APPLICATION CLASS
# ============================================================================

class NeuronApp(QMainWindow):
    """
    Main Neuron Application Class

    Complete electrophysiology platform for:
    - Real-time data acquisition (SWP, EEG, MEA modes)
    - Offline data analysis
    - Spike detection and sorting
    - Signal processing and filtering
    - Statistical analysis
    - Event detection
    - Protocol-based experiments
    """

    def __init__(self):
        super().__init__()

        # Version information
        self.version = 1020000

        # Initialize data structures
        self.colors = NeuronColors()
        self.meting = Meting()
        self.spikes = SpikeData()
        self.events = EventData()
        self.axon = AxonAmplifier()
        self.timeline = Timeline()
        self.mea_data = None

        # Application state
        self.online = False
        self.abort = False
        self.always_abort = False
        self.nrn_abort = False
        self.nrn_busy = False
        self.in_config = False
        self.mode = AcquisitionMode.OFFLINE

        # GUI state
        self.zoom = Zoom()
        self.cursor = Cursor()
        self.video = VideoSettings()
        self.file_settings = FileSettings()

        # Time window state for navigation
        self.time_start = 0.0  # Start time in samples
        self.time_end = 1000.0  # End time in samples
        self.time_duration = 1000.0  # Duration in samples
        self.sampling_rate = 1000.0  # Default sampling rate in Hz

        # Timers
        self.timers = {}

        # DAQ devices
        self.daq_devices: List[DAQDevice] = []

        # Memory management
        self.memory_set = []
        self.cache = {'status': False}

        # Data storage
        self.current_file_nr = 0
        self.current_seg_nr = 0

        # Initialize UI
        self.init_ui()

        # Initialize subsystems
        self.reset_all(ask=False)


    # ========================================================================
    # INITIALIZATION AND CONFIGURATION
    # ========================================================================

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle(f"Neuron ({self.get_version():.1f})")
        self.setGeometry(100, 100, 1600, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget for different panels
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create panels
        self.create_cmd_panel()
        self.create_display_panel()
        self.create_analysis_panel()
        self.create_config_panel()

        # Status bar
        self.statusBar().showMessage('Ready')


    def create_cmd_panel(self):
        """Create command control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File controls
        file_group = QGroupBox("File Control")
        file_layout = QHBoxLayout()

        self.btn_load_data = QPushButton("Load Data")
        self.btn_load_data.clicked.connect(self.load_data)
        file_layout.addWidget(self.btn_load_data)

        self.btn_save_data = QPushButton("Save Data")
        self.btn_save_data.clicked.connect(self.save_neuron)
        file_layout.addWidget(self.btn_save_data)

        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.clicked.connect(self.prev_file)
        file_layout.addWidget(self.btn_prev_file)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.clicked.connect(self.next_file)
        file_layout.addWidget(self.btn_next_file)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Acquisition controls
        acq_group = QGroupBox("Acquisition")
        acq_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Offline", "SWP", "EEG", "TST", "MEA"])
        self.combo_mode.currentIndexChanged.connect(self.change_mode)
        mode_layout.addWidget(self.combo_mode)
        acq_layout.addLayout(mode_layout)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_acquisition)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_acquisition)
        btn_layout.addWidget(self.btn_stop)

        acq_layout.addLayout(btn_layout)
        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)

        # Zoom controls
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()

        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.btn_zoom_out)

        self.btn_zoom_reset = QPushButton("Reset")
        self.btn_zoom_reset.clicked.connect(self.zoom_reset)
        zoom_layout.addWidget(self.btn_zoom_reset)

        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        self.tab_widget.addTab(panel, "Command")


    def create_display_panel(self):
        """Create data display panel with NEURON-style navigation"""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(3)

        # Timeline overview bar (shows full data extent)
        timeline_frame = QWidget()
        timeline_frame.setFixedHeight(40)
        timeline_frame.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        timeline_layout = QVBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(5, 5, 5, 5)

        self.timeline_label = QLabel("Timeline: 0.00s - 0.00s (Total: 0.00s)")
        self.timeline_label.setStyleSheet("border: none; background: transparent;")
        timeline_layout.addWidget(self.timeline_label)

        # Timeline position indicator (shows current view window)
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setFixedHeight(20)
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_changed)
        self.timeline_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                border: 1px solid #5c5c5c;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
        """)
        timeline_layout.addWidget(self.timeline_slider)

        main_layout.addWidget(timeline_frame)

        # Main plot canvas with overview and detail axes
        self.figure = Figure(figsize=(12, 7))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(450)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Enable mousewheel events for zoom
        self.canvas.mpl_connect('scroll_event', self.on_mousewheel_zoom)

        # Enable click events on overview for navigation
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        main_layout.addWidget(self.canvas, stretch=10)

        # Compact navigation controls
        nav_container = QWidget()
        nav_container.setMaximumHeight(100)
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(3)

        # Compact navigation row
        nav_row1 = QHBoxLayout()

        # Time window display (read-only, more compact)
        nav_row1.addWidget(QLabel("Window:"))
        self.spin_time_duration = QDoubleSpinBox()
        self.spin_time_duration.setDecimals(2)
        self.spin_time_duration.setSuffix(" s")
        self.spin_time_duration.setValue(1.0)
        self.spin_time_duration.setMinimum(0.01)
        self.spin_time_duration.setMaximum(1000)
        self.spin_time_duration.setMaximumWidth(100)
        self.spin_time_duration.valueChanged.connect(self.on_time_duration_changed)
        nav_row1.addWidget(self.spin_time_duration)

        nav_row1.addSpacing(20)

        # Navigation buttons (compact)
        self.btn_jump_start = QPushButton("|◀")
        self.btn_jump_start.setToolTip("Jump to start (Home)")
        self.btn_jump_start.setMaximumWidth(40)
        self.btn_jump_start.clicked.connect(self.jump_to_start)
        nav_row1.addWidget(self.btn_jump_start)

        self.btn_backward_full = QPushButton("◀◀")
        self.btn_backward_full.setToolTip("Backward full (PgUp)")
        self.btn_backward_full.setMaximumWidth(40)
        self.btn_backward_full.clicked.connect(self.backward_full)
        nav_row1.addWidget(self.btn_backward_full)

        self.btn_backward_half = QPushButton("◀")
        self.btn_backward_half.setToolTip("Backward half (Left)")
        self.btn_backward_half.setMaximumWidth(40)
        self.btn_backward_half.clicked.connect(self.backward_half)
        nav_row1.addWidget(self.btn_backward_half)

        self.btn_forward_half = QPushButton("▶")
        self.btn_forward_half.setToolTip("Forward half (Right)")
        self.btn_forward_half.setMaximumWidth(40)
        self.btn_forward_half.clicked.connect(self.forward_half)
        nav_row1.addWidget(self.btn_forward_half)

        self.btn_forward_full = QPushButton("▶▶")
        self.btn_forward_full.setToolTip("Forward full (PgDn)")
        self.btn_forward_full.setMaximumWidth(40)
        self.btn_forward_full.clicked.connect(self.forward_full)
        nav_row1.addWidget(self.btn_forward_full)

        self.btn_jump_end = QPushButton("▶|")
        self.btn_jump_end.setToolTip("Jump to end (End)")
        self.btn_jump_end.setMaximumWidth(40)
        self.btn_jump_end.clicked.connect(self.jump_to_end)
        nav_row1.addWidget(self.btn_jump_end)

        nav_row1.addSpacing(20)

        # Zoom buttons
        nav_row1.addWidget(QLabel("Zoom:"))
        self.btn_zoom_in = QPushButton("In [+]")
        self.btn_zoom_in.setMaximumWidth(60)
        self.btn_zoom_in.setToolTip("Zoom in (Mouse wheel up)")
        self.btn_zoom_in.clicked.connect(self.zoom_in_time)
        nav_row1.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("Out [-]")
        self.btn_zoom_out.setMaximumWidth(60)
        self.btn_zoom_out.setToolTip("Zoom out (Mouse wheel down)")
        self.btn_zoom_out.clicked.connect(self.zoom_out_time)
        nav_row1.addWidget(self.btn_zoom_out)

        nav_row1.addStretch()
        nav_layout.addLayout(nav_row1)

        main_layout.addWidget(nav_container)

        # Display controls - more compact
        display_group = QGroupBox("Display Settings")
        display_group.setMaximumHeight(80)  # Limit display group height
        display_layout = QHBoxLayout()

        display_layout.addWidget(QLabel("Channel:"))
        self.spin_channel = QSpinBox()
        self.spin_channel.setMinimum(0)
        self.spin_channel.setMaximum(128)
        self.spin_channel.setSpecialValueText("All")
        self.spin_channel.valueChanged.connect(self.redraw)
        display_layout.addWidget(self.spin_channel)

        display_layout.addWidget(QLabel("Gain:"))
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setDecimals(2)
        self.spin_gain.setMinimum(0.01)
        self.spin_gain.setMaximum(1000)
        self.spin_gain.setValue(1.0)
        self.spin_gain.valueChanged.connect(self.redraw)
        display_layout.addWidget(self.spin_gain)

        display_layout.addWidget(QLabel("Separation:"))
        self.spin_separation = QDoubleSpinBox()
        self.spin_separation.setDecimals(2)
        self.spin_separation.setMinimum(0.5)
        self.spin_separation.setMaximum(5.0)
        self.spin_separation.setValue(1.5)
        self.spin_separation.setToolTip("Vertical spacing between channels")
        self.spin_separation.valueChanged.connect(self.redraw)
        display_layout.addWidget(self.spin_separation)

        display_layout.addWidget(QLabel("Max Channels:"))
        self.spin_max_channels = QSpinBox()
        self.spin_max_channels.setMinimum(1)
        self.spin_max_channels.setMaximum(256)
        self.spin_max_channels.setValue(72)
        self.spin_max_channels.setToolTip("Maximum number of channels to display in multi-channel mode")
        self.spin_max_channels.valueChanged.connect(self.redraw)
        display_layout.addWidget(self.spin_max_channels)

        display_layout.addStretch()
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)

        self.tab_widget.addTab(panel, "Display")


    def create_analysis_panel(self):
        """Create analysis panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Analysis tools
        tools_group = QGroupBox("Analysis Tools")
        tools_layout = QVBoxLayout()

        btn_spikes = QPushButton("Spike Detection")
        btn_spikes.clicked.connect(self.do_spikes)
        tools_layout.addWidget(btn_spikes)

        btn_template = QPushButton("Template Matching")
        btn_template.clicked.connect(self.do_template)
        tools_layout.addWidget(btn_template)

        btn_filter = QPushButton("Filter")
        btn_filter.clicked.connect(self.do_filter)
        tools_layout.addWidget(btn_filter)

        btn_statistics = QPushButton("Statistics")
        btn_statistics.clicked.connect(self.do_statistics)
        tools_layout.addWidget(btn_statistics)

        btn_spectrum = QPushButton("Power Spectrum")
        btn_spectrum.clicked.connect(self.show_spectrum)
        tools_layout.addWidget(btn_spectrum)

        btn_psth = QPushButton("PSTH")
        btn_psth.clicked.connect(self.do_psth)
        tools_layout.addWidget(btn_psth)

        btn_minis = QPushButton("Mini Analysis")
        btn_minis.clicked.connect(self.do_minis)
        tools_layout.addWidget(btn_minis)

        tools_layout.addStretch()
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.tab_widget.addTab(panel, "Analysis")


    def create_config_panel(self):
        """Create configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # DAQ Configuration
        daq_group = QGroupBox("DAQ Configuration")
        daq_layout = QVBoxLayout()

        btn_daq_config = QPushButton("Configure DAQ")
        btn_daq_config.clicked.connect(self.configure_daq)
        daq_layout.addWidget(btn_daq_config)

        btn_show_daqs = QPushButton("Show DAQ Devices")
        btn_show_daqs.clicked.connect(self.show_daqs)
        daq_layout.addWidget(btn_show_daqs)

        daq_group.setLayout(daq_layout)
        layout.addWidget(daq_group)

        # Protocol configuration
        protocol_group = QGroupBox("Protocol")
        protocol_layout = QVBoxLayout()

        btn_load_protocol = QPushButton("Load Protocol")
        btn_load_protocol.clicked.connect(self.load_protocol)
        protocol_layout.addWidget(btn_load_protocol)

        btn_save_protocol = QPushButton("Save Protocol")
        btn_save_protocol.clicked.connect(self.save_protocol)
        protocol_layout.addWidget(btn_save_protocol)

        protocol_group.setLayout(protocol_layout)
        layout.addWidget(protocol_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        btn_reset = QPushButton("Reset All")
        btn_reset.clicked.connect(lambda: self.reset_all(ask=True))
        settings_layout.addWidget(btn_reset)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        layout.addStretch()

        self.tab_widget.addTab(panel, "Config")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation"""
        from PyQt5.QtCore import Qt
        key = event.key()

        if key == Qt.Key_Left:
            self.backward_half()
        elif key == Qt.Key_Right:
            self.forward_half()
        elif key == Qt.Key_PageUp:
            self.backward_full()
        elif key == Qt.Key_PageDown:
            self.forward_full()
        elif key == Qt.Key_Home:
            self.jump_to_start()
        elif key == Qt.Key_End:
            self.jump_to_end()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.zoom_in_time()
        elif key == Qt.Key_Minus or key == Qt.Key_Underscore:
            self.zoom_out_time()
        else:
            super().keyPressEvent(event)

    def reset_all(self, ask=True):
        """Reset entire application state"""
        if ask:
            reply = QMessageBox.question(self, 'Reset DAQ?',
                                       'Reset all settings?',
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Stop all timers
        for timer in self.timers.values():
            if timer.isActive():
                timer.stop()
        self.timers.clear()

        # Reset state
        self.online = False
        self.nrn_abort = False
        self.nrn_busy = False
        self.abort = False
        self.always_abort = False

        # Reset data structures
        self.meting = Meting()
        self.spikes = SpikeData()
        self.events = EventData()
        self.timeline = Timeline()

        # Initialize subsystems
        self.daq_define()
        self.template_ini()
        self.trigger_ini()
        self.spike_ini()

        self.statusBar().showMessage('System reset complete')


    # ========================================================================
    # VERSION AND UTILITY FUNCTIONS
    # ========================================================================

    def get_version(self) -> float:
        """Get version number"""
        return self.version / 1000.0


    def set_menu(self, txt: str):
        """Set window title/menu text"""
        if txt == 'version':
            txt = f"Neuron ({self.get_version():.1f})"
        elif txt == 'error':
            txt = f"Neuron error: {txt}"
        else:
            txt = f"Neuron - {txt}"
        self.setWindowTitle(txt)


    def message(self, txt: str):
        """Show message dialog"""
        QMessageBox.information(self, "Message", txt)


    def waitbar(self, fraction: float, txt: str):
        """Update progress bar"""
        self.progress_bar.setValue(int(fraction * 100))
        self.progress_bar.setFormat(txt)
        QApplication.processEvents()


    def spk_color(self, index: int) -> Tuple[float, float, float]:
        """Get spike waveform color by index"""
        colors = [
            (0.8, 0.95, 0.95),
            (0.95, 0.8, 0.95),
            (0.95, 0.95, 0.8),
            (0.8, 0.80, 0.95),
            (0.95, 0.8, 0.80),
            (0.8, 0.95, 0.80)
        ]
        return colors[index % 6]


    def unquote(self, s: str) -> str:
        """Remove quotes and whitespace from string"""
        s = s.strip()
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
            s = s[1:-1]
        return s


    def unifile(self, path: str) -> str:
        """Uniform file path for current OS"""
        return str(Path(path))


    # ========================================================================
    # TIME NAVIGATION FUNCTIONS
    # ========================================================================

    def update_time_window(self, start=None, end=None, duration=None):
        """Update time window parameters and timeline display"""
        total_samples = len(self.meting.adc) if len(self.meting.adc) > 0 else 1000

        if start is not None:
            self.time_start = max(0, min(start, total_samples))
        if duration is not None:
            self.time_duration = max(10, min(duration, total_samples))
        if end is not None:
            self.time_end = max(self.time_duration, min(end, total_samples))

        # Ensure consistency
        if start is not None and duration is not None:
            self.time_end = min(self.time_start + self.time_duration, total_samples)
        elif start is not None and end is not None:
            self.time_duration = self.time_end - self.time_start
        elif duration is not None and end is not None:
            self.time_start = max(0, self.time_end - self.time_duration)

        # Update duration spinbox (block signals to avoid recursion)
        self.spin_time_duration.blockSignals(True)
        self.spin_time_duration.setValue(self.time_duration / self.sampling_rate)
        self.spin_time_duration.blockSignals(False)

        # Update timeline label
        start_time = self.time_start / self.sampling_rate
        end_time = self.time_end / self.sampling_rate
        total_time = total_samples / self.sampling_rate
        self.timeline_label.setText(f"Timeline: {start_time:.2f}s - {end_time:.2f}s (Total: {total_time:.2f}s)")

        # Update timeline slider
        if total_samples > self.time_duration:
            slider_pos = int(1000 * self.time_start / (total_samples - self.time_duration))
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(slider_pos)
            self.timeline_slider.blockSignals(False)

    def on_time_duration_changed(self, value):
        """Handle duration change"""
        duration_samples = value * self.sampling_rate
        self.update_time_window(duration=duration_samples)
        self.redraw()

    def on_timeline_slider_changed(self, value):
        """Handle timeline slider change - memory efficient scrolling"""
        total_samples = len(self.meting.adc) if len(self.meting.adc) > 0 else 1000
        if total_samples > self.time_duration:
            # Calculate start position from slider value (0-1000 range)
            start = (value / 1000.0) * (total_samples - self.time_duration)
            self.update_time_window(start=start)
            self.redraw()

    def on_mousewheel_zoom(self, event):
        """Handle mousewheel zoom - zoom in/out at cursor position"""
        if event.inaxes and event.inaxes.get_label() == 'detail':
            # Get cursor position in data coordinates
            xdata = event.xdata

            # Zoom factor
            zoom_factor = 0.8 if event.button == 'up' else 1.25

            # Calculate new duration
            new_duration = self.time_duration * zoom_factor
            new_duration = max(10, min(new_duration, len(self.meting.adc)))

            # Zoom centered on cursor position
            cursor_samples = xdata * self.sampling_rate
            relative_pos = (cursor_samples - self.time_start) / self.time_duration
            new_start = cursor_samples - new_duration * relative_pos

            self.update_time_window(start=new_start, duration=new_duration)
            self.redraw()

    def on_canvas_click(self, event):
        """Handle click on canvas - navigate via overview"""
        if event.inaxes and event.inaxes.get_label() == 'overview':
            # Click on overview - jump to that position
            if len(self.meting.adc) > 0:
                click_time = event.xdata  # In seconds
                click_sample = click_time * self.sampling_rate

                # Center window on clicked position
                new_start = click_sample - self.time_duration / 2
                self.update_time_window(start=new_start)
                self.redraw()

    def jump_to_start(self):
        """Jump to start of recording"""
        self.update_time_window(start=0)
        self.redraw()

    def jump_to_end(self):
        """Jump to end of recording"""
        total_samples = len(self.meting.adc) if len(self.meting.adc) > 0 else 1000
        self.update_time_window(start=total_samples - self.time_duration)
        self.redraw()

    def forward_half(self):
        """Move forward by half duration"""
        step = self.time_duration * 0.5
        self.update_time_window(start=self.time_start + step)
        self.redraw()

    def forward_full(self):
        """Move forward by full duration"""
        self.update_time_window(start=self.time_start + self.time_duration)
        self.redraw()

    def backward_half(self):
        """Move backward by half duration"""
        step = self.time_duration * 0.5
        self.update_time_window(start=self.time_start - step)
        self.redraw()

    def backward_full(self):
        """Move backward by full duration"""
        self.update_time_window(start=self.time_start - self.time_duration)
        self.redraw()

    def zoom_in_time(self):
        """Zoom in (reduce visible duration by 50%)"""
        new_duration = self.time_duration * 0.5
        center = self.time_start + self.time_duration * 0.5
        self.update_time_window(start=center - new_duration * 0.5, duration=new_duration)
        self.redraw()

    def zoom_out_time(self):
        """Zoom out (increase visible duration by 100%)"""
        new_duration = self.time_duration * 2.0
        center = self.time_start + self.time_duration * 0.5
        self.update_time_window(start=center - new_duration * 0.5, duration=new_duration)
        self.redraw()

    # ========================================================================
    # DAQ HARDWARE LAYER
    # ========================================================================

    def daq_define(self):
        """Define available DAQ devices"""
        # Placeholder for DAQ device detection
        # In real implementation, would detect NI-DAQ, MCC, etc.
        self.daq_devices = []

        # Example device
        device = DAQDevice(
            type='nidaq',
            BoardName='Dev1',
            InstalledBoardId=1,
            adc=list(range(8)),
            dac=list(range(2))
        )
        self.daq_devices.append(device)

        self.statusBar().showMessage(f'Detected {len(self.daq_devices)} DAQ device(s)')


    def show_daqs(self):
        """Display DAQ device information"""
        info = "DAQ Devices:\n\n"
        for i, dev in enumerate(self.daq_devices):
            info += f"Device {i+1}:\n"
            info += f"  Type: {dev.type}\n"
            info += f"  Name: {dev.BoardName}\n"
            info += f"  ADC channels: {len(dev.adc)}\n"
            info += f"  DAC channels: {len(dev.dac)}\n\n"

        self.message(info)


    def configure_daq(self):
        """Open DAQ configuration dialog"""
        self.message("DAQ configuration dialog\n(Not implemented in this demo)")


    def adc_ini(self, device_idx: int, adc_type: str = 'default'):
        """Initialize ADC channels"""
        if device_idx >= len(self.daq_devices):
            return

        device = self.daq_devices[device_idx]

        # Create ADC channel configurations
        for i, adc_hw in enumerate(device.adc):
            channel = ADCChannel(
                device=device_idx,
                local=i,
                hwchan=adc_hw,
                gain=1.0,
                usermax=100.0,
                name=f'adc{i}',
                unit='mV',
                junction=0.0,
                color='k'
            )
            self.meting.ADC.append(channel)


    def dac_ini(self, device_idx: int):
        """Initialize DAC channels"""
        if device_idx >= len(self.daq_devices):
            return

        device = self.daq_devices[device_idx]

        # Create DAC channel configurations
        for i, dac_hw in enumerate(device.dac):
            channel = DACChannel(
                device=device_idx,
                local=i,
                hwchan=dac_hw,
                usermax=100.0,
                name=f'dac{i}',
                unit='mV',
                junction=0.0,
                color='k'
            )
            self.meting.DAC.append(channel)


    # ========================================================================
    # FILE I/O OPERATIONS
    # ========================================================================

    def load_data(self):
        """Load measurement data from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Data",
            self.file_settings.root,
            "EEG files (*.dat *.mat);;MAT files (*.mat);;DAT files (*.dat);;HDF5 files (*.h5);;All files (*.*)"
        )

        if not filename:
            return

        try:
            if filename.endswith('.dat'):
                self.load_dat_file(filename)
            elif filename.endswith('.mat'):
                self.load_mat_file(filename)
            elif filename.endswith('.h5'):
                self.load_h5_file(filename)
            else:
                self.message("Unsupported file format")
                return

            self.statusBar().showMessage(f'Loaded: {Path(filename).name}')
            self.redraw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")


    def load_dat_file(self, filename: str):
        """Load .dat file containing multichannel EEG data"""
        if EEGLoader is None:
            raise ImportError("eeg_loader.py not found. Cannot load .dat files.")

        # Try to infer channel count, or use common values
        num_channels = None

        # Check if there's a companion file with channel info
        metadata_file = Path(filename).parent / "channel_info.txt"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    for line in f:
                        if 'channels' in line.lower():
                            num_channels = int(line.split(':')[1].strip())
                            break
            except:
                pass

        # Initialize loader
        try:
            loader = EEGLoader(filename, num_channels=num_channels)
        except Exception as e:
            # Try common channel counts
            for nch in [16, 32, 64, 72, 128]:
                try:
                    loader = EEGLoader(filename, num_channels=nch)
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not determine channel count for {filename}")

        # Load all data (for now - could add chunking for very large files)
        # Load first 10 seconds or 100k samples, whichever is smaller
        max_samples = min(100000, loader.num_samples_per_channel)

        self.meting.adc = loader.load_all_channels(start_sample=0, end_sample=max_samples, dtype=np.float32)

        # Store sampling rate if available
        if loader.timestamps is not None and len(loader.timestamps) > 1:
            sampling_rate = 1.0 / np.mean(np.diff(loader.timestamps))
        else:
            sampling_rate = 1000.0  # Default 1kHz

        # Create ADC channel metadata
        self.meting.ADC = []
        for i in range(loader.num_channels):
            channel = ADCChannel(
                device=0,
                local=i,
                hwchan=i,
                gain=1.0,
                usermax=1000.0,
                name=f'EEG{i+1}',
                unit='μV',
                junction=0.0,
                color='k'
            )
            self.meting.ADC.append(channel)

        # Initialize time window
        self.sampling_rate = sampling_rate
        self.time_start = 0
        self.time_duration = min(int(sampling_rate), max_samples)  # 1 second or less
        self.time_end = self.time_duration

        # Update time controls
        self.spin_time_duration.setMaximum(max_samples / sampling_rate)
        self.update_time_window(start=0, duration=self.time_duration)

        self.statusBar().showMessage(
            f'Loaded {loader.num_channels} channels, {max_samples} samples @ {sampling_rate:.1f} Hz'
        )

    def load_mat_file(self, filename: str):
        """Load MATLAB .mat file containing EEG data or Meting structure"""
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("scipy is required to load .mat files")

        mat_data = sio.loadmat(filename)

        # Check if this is a Neuron.m Meting structure
        if 'Meting' in mat_data:
            # Load Neuron.m format
            meting_struct = mat_data['Meting']

            # MATLAB structures are stored as structured arrays
            if meting_struct.dtype.names:
                # Extract ADC data
                if 'adc' in meting_struct.dtype.names:
                    self.meting.adc = meting_struct['adc'][0, 0].astype(np.float32)

                # Extract DAC data
                if 'dac' in meting_struct.dtype.names:
                    self.meting.dac = meting_struct['dac'][0, 0]

                # Extract metadata
                if 'serienaam' in meting_struct.dtype.names:
                    try:
                        self.meting.serienaam = str(meting_struct['serienaam'][0, 0][0])
                    except:
                        pass

        # Check for raw EEG data (similar format to .dat)
        elif 'data' in mat_data:
            self.meting.adc = mat_data['data'].astype(np.float32)

        # Check for direct adc/dac format
        elif 'adc' in mat_data:
            self.meting.adc = mat_data['adc'].astype(np.float32)
            if 'dac' in mat_data:
                self.meting.dac = mat_data['dac']

        # If no recognized format, try to load any large array
        else:
            # Find the largest array that's not metadata
            largest_key = None
            largest_size = 0
            for key in mat_data.keys():
                if not key.startswith('__'):
                    arr = mat_data[key]
                    if isinstance(arr, np.ndarray) and arr.size > largest_size:
                        largest_key = key
                        largest_size = arr.size

            if largest_key:
                self.meting.adc = mat_data[largest_key].astype(np.float32)
                self.statusBar().showMessage(f'Loaded array "{largest_key}" from .mat file')
            else:
                raise ValueError("No suitable data array found in .mat file")

        # Ensure adc is 2D (samples x channels)
        if len(self.meting.adc.shape) == 1:
            self.meting.adc = self.meting.adc.reshape(-1, 1)
        elif len(self.meting.adc.shape) == 3:
            # If 3D (samples x channels x sweeps), reshape or take first sweep
            self.meting.adc = self.meting.adc[:, :, 0]

        # Create channel metadata if not present
        if len(self.meting.ADC) == 0:
            num_channels = self.meting.adc.shape[1]
            for i in range(num_channels):
                channel = ADCChannel(
                    device=0,
                    local=i,
                    hwchan=i,
                    gain=1.0,
                    usermax=1000.0,
                    name=f'Ch{i+1}',
                    unit='a.u.',
                    junction=0.0,
                    color='k'
                )
                self.meting.ADC.append(channel)

        # Initialize time window after loading MAT file
        num_samples = self.meting.adc.shape[0]
        self.sampling_rate = 1000.0  # Default 1kHz, adjust if metadata available
        self.time_start = 0
        self.time_duration = min(int(self.sampling_rate), num_samples)  # 1 second
        self.time_end = self.time_duration

        # Update time controls
        self.spin_time_duration.setMaximum(num_samples / self.sampling_rate)
        self.update_time_window(start=0, duration=self.time_duration)


    def load_h5_file(self, filename: str):
        """Load HDF5 file (MEA data)"""
        with h5py.File(filename, 'r') as f:
            # Extract MEA data structure
            if 'data' in f:
                data = f['data'][:]
                self.mea_data = MEAData(data=data)
                self.meting.adc = data


    def save_neuron(self):
        """Save measurement data"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Data",
            self.file_settings.root,
            "MAT files (*.mat);;HDF5 files (*.h5);;All files (*.*)"
        )

        if not filename:
            return

        try:
            if filename.endswith('.mat'):
                self.save_mat_file(filename)
            elif filename.endswith('.h5'):
                self.save_h5_file(filename)
            else:
                filename += '.mat'
                self.save_mat_file(filename)

            self.statusBar().showMessage(f'Saved: {filename}')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")


    def save_mat_file(self, filename: str):
        """Save as MATLAB .mat file"""
        import scipy.io as sio

        save_dict = {
            'adc': self.meting.adc,
            'dac': self.meting.dac,
            'datetime': self.meting.datetime,
            'protocol': self.meting.protocol
        }

        sio.savemat(filename, save_dict)


    def save_h5_file(self, filename: str):
        """Save as HDF5 file"""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('adc', data=self.meting.adc)
            if self.meting.dac is not None and len(self.meting.dac) > 0:
                f.create_dataset('dac', data=self.meting.dac)


    def prev_file(self):
        """Load previous file"""
        self.current_file_nr -= 1
        self.statusBar().showMessage(f'File #{self.current_file_nr}')


    def next_file(self):
        """Load next file"""
        self.current_file_nr += 1
        self.statusBar().showMessage(f'File #{self.current_file_nr}')


    # ========================================================================
    # ACQUISITION MODE CONTROL
    # ========================================================================

    def change_mode(self, index: int):
        """Change acquisition mode"""
        modes = [AcquisitionMode.OFFLINE, AcquisitionMode.SWP,
                AcquisitionMode.EEG, AcquisitionMode.TST, AcquisitionMode.MEA]
        self.mode = modes[index]
        self.statusBar().showMessage(f'Mode: {self.mode.name}')


    def start_acquisition(self):
        """Start data acquisition"""
        if self.mode == AcquisitionMode.SWP:
            self.swp_run()
        elif self.mode == AcquisitionMode.EEG:
            self.eeg_run()
        elif self.mode == AcquisitionMode.TST:
            self.tst_run()
        else:
            self.message("Select acquisition mode first")


    def stop_acquisition(self):
        """Stop data acquisition"""
        self.abort = True

        # Stop timers
        for timer in self.timers.values():
            if timer.isActive():
                timer.stop()

        self.statusBar().showMessage('Acquisition stopped')


    def swp_run(self):
        """Run sweep acquisition"""
        self.online = True
        self.nrn_busy = True

        # Create timer for sweep updates
        if 'swp' not in self.timers:
            self.timers['swp'] = QTimer()
            self.timers['swp'].timeout.connect(self.swp_timer)

        self.timers['swp'].start(50)  # 50ms updates
        self.statusBar().showMessage('SWP acquisition running...')


    def swp_timer(self):
        """Sweep timer callback"""
        if self.abort:
            self.timers['swp'].stop()
            self.nrn_busy = False
            return

        # Acquire data (placeholder - would interface with actual DAQ)
        # For demo, generate synthetic data
        if len(self.meting.adc) == 0:
            self.meting.adc = np.random.randn(10000) * 10

        # Update display
        self.redraw()


    def eeg_run(self):
        """Run continuous EEG acquisition"""
        self.online = True
        self.nrn_busy = True

        # Create timer for EEG updates
        if 'eeg' not in self.timers:
            self.timers['eeg'] = QTimer()
            self.timers['eeg'].timeout.connect(self.eeg_timer)

        self.timers['eeg'].start(100)  # 100ms updates
        self.statusBar().showMessage('EEG acquisition running...')


    def eeg_timer(self):
        """EEG timer callback"""
        if self.abort:
            self.timers['eeg'].stop()
            self.nrn_busy = False
            return

        # Continuous data acquisition
        # Placeholder implementation
        pass


    def tst_run(self):
        """Run test mode"""
        self.online = True
        self.statusBar().showMessage('Test mode')


    # ========================================================================
    # SIGNAL PROCESSING
    # ========================================================================

    def do_filter(self):
        """Apply filter to data"""
        if len(self.meting.adc) == 0:
            self.message("No data loaded")
            return

        # Simple lowpass filter example
        fs = 10000  # Sampling frequency
        fc = 1000   # Cutoff frequency

        b, a = signal.butter(4, fc/(fs/2), 'low')

        if self.meting.adc.ndim == 1:
            filtered = signal.filtfilt(b, a, self.meting.adc)
        else:
            filtered = np.zeros_like(self.meting.adc)
            for i in range(self.meting.adc.shape[1]):
                filtered[:, i] = signal.filtfilt(b, a, self.meting.adc[:, i])

        self.meting.adc = filtered
        self.redraw()
        self.message(f"Applied lowpass filter (fc={fc} Hz)")


    def filter_trace(self, data: np.ndarray, filter_type: str = 'lowpass',
                     cutoff: float = 1000, fs: float = 10000, order: int = 4) -> np.ndarray:
        """Apply filter to trace"""
        nyq = fs / 2
        normalized_cutoff = cutoff / nyq

        if filter_type == 'lowpass':
            b, a = signal.butter(order, normalized_cutoff, 'low')
        elif filter_type == 'highpass':
            b, a = signal.butter(order, normalized_cutoff, 'high')
        elif filter_type == 'bandpass':
            b, a = signal.butter(order, normalized_cutoff, 'bandpass')
        else:
            return data

        return signal.filtfilt(b, a, data)


    def do_clean_50hz(self):
        """Remove 50Hz line noise"""
        if len(self.meting.adc) == 0:
            return

        fs = 10000  # Sampling frequency

        # Notch filter at 50Hz
        b, a = signal.iirnotch(50, 30, fs)

        if self.meting.adc.ndim == 1:
            cleaned = signal.filtfilt(b, a, self.meting.adc)
        else:
            cleaned = np.zeros_like(self.meting.adc)
            for i in range(self.meting.adc.shape[1]):
                cleaned[:, i] = signal.filtfilt(b, a, self.meting.adc[:, i])

        self.meting.adc = cleaned
        self.redraw()


    def do_resample(self, new_rate: float):
        """Resample data to new sampling rate"""
        if len(self.meting.adc) == 0:
            return

        old_rate = 10000  # Current sampling rate
        ratio = new_rate / old_rate
        new_length = int(len(self.meting.adc) * ratio)

        if self.meting.adc.ndim == 1:
            resampled = signal.resample(self.meting.adc, new_length)
        else:
            resampled = np.zeros((new_length, self.meting.adc.shape[1]))
            for i in range(self.meting.adc.shape[1]):
                resampled[:, i] = signal.resample(self.meting.adc[:, i], new_length)

        self.meting.adc = resampled


    # ========================================================================
    # SPIKE DETECTION AND ANALYSIS
    # ========================================================================

    def spike_ini(self):
        """Initialize spike detection"""
        self.spikes = SpikeData()


    def template_ini(self):
        """Initialize template matching"""
        self.spikes.template = None
        self.spikes.tmpnul = None
        self.spikes.tmprng = None


    def trigger_ini(self):
        """Initialize trigger detection"""
        pass


    def do_spikes(self):
        """Detect spikes in data"""
        if len(self.meting.adc) == 0:
            self.message("No data loaded")
            return

        data = self.meting.adc if self.meting.adc.ndim == 1 else self.meting.adc[:, 0]

        # Simple threshold-based spike detection
        threshold = np.std(data) * 3

        # Find threshold crossings
        crossings = self.find_crossings(data, threshold)

        results = f"Spike Detection Results:\n\n"
        results += f"Threshold: {threshold:.2f}\n"
        results += f"Number of spikes: {len(crossings)}\n"

        if len(crossings) > 0:
            isi = np.diff(crossings)
            results += f"Mean ISI: {np.mean(isi):.2f} samples\n"
            results += f"Firing rate: {len(crossings) / (len(data)/10000):.2f} Hz\n"

        self.results_text.setText(results)
        self.statusBar().showMessage(f'Detected {len(crossings)} spikes')


    def find_crossings(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Find threshold crossings"""
        above = data > threshold
        crossings = np.where(np.diff(above.astype(int)) > 0)[0]
        return crossings


    def do_template(self):
        """Template matching spike detection"""
        if len(self.meting.adc) == 0:
            self.message("No data loaded")
            return

        # Placeholder for template matching
        self.message("Template matching\n(Advanced implementation required)")


    def get_peaks(self, data: np.ndarray, min_distance: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks in data"""
        peaks, properties = signal.find_peaks(data, distance=min_distance)
        return peaks, data[peaks]


    def spike_width(self, data: np.ndarray, spike_idx: int,
                   threshold_fraction: float = 0.5) -> float:
        """Calculate spike width at threshold fraction of amplitude"""
        if spike_idx >= len(data):
            return 0

        peak_val = data[spike_idx]
        threshold = peak_val * threshold_fraction

        # Find width at threshold
        left = spike_idx
        while left > 0 and data[left] > threshold:
            left -= 1

        right = spike_idx
        while right < len(data) - 1 and data[right] > threshold:
            right += 1

        return right - left


    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================

    def do_statistics(self):
        """Perform statistical analysis"""
        if len(self.meting.adc) == 0:
            self.message("No data loaded")
            return

        data = self.meting.adc if self.meting.adc.ndim == 1 else self.meting.adc[:, 0]

        results = "Statistical Analysis:\n\n"
        results += f"Mean: {np.mean(data):.4f}\n"
        results += f"Std Dev: {np.std(data):.4f}\n"
        results += f"Median: {np.median(data):.4f}\n"
        results += f"Min: {np.min(data):.4f}\n"
        results += f"Max: {np.max(data):.4f}\n"
        results += f"RMS: {np.sqrt(np.mean(data**2)):.4f}\n"

        # Percentiles
        p25, p75 = np.percentile(data, [25, 75])
        results += f"25th percentile: {p25:.4f}\n"
        results += f"75th percentile: {p75:.4f}\n"

        # Skewness and kurtosis
        results += f"Skewness: {stats.skew(data):.4f}\n"
        results += f"Kurtosis: {stats.kurtosis(data):.4f}\n"

        self.results_text.setText(results)


    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def redraw(self):
        """Redraw data display - multichannel EEG with overview and detail view"""
        if len(self.meting.adc) == 0:
            return

        self.figure.clear()

        full_data = self.meting.adc
        num_samples_total = len(full_data)

        # Get display parameters
        gain = self.spin_gain.value()
        channel_to_show = self.spin_channel.value()  # 0 = all channels
        separation_factor = self.spin_separation.value()

        # Determine channels to display
        if full_data.ndim == 1 or full_data.shape[1] == 1:
            single_channel_mode = True
            channels_to_plot = [0]
        else:
            num_channels = full_data.shape[1]
            if channel_to_show > 0 and channel_to_show <= num_channels:
                single_channel_mode = True
                channels_to_plot = [channel_to_show - 1]
            else:
                single_channel_mode = False
                max_channels_to_display = min(num_channels, self.spin_max_channels.value())
                channels_to_plot = list(range(max_channels_to_display))

        # Create subplot grid: overview on top (15%), detail below (85%)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.25)
        ax_overview = self.figure.add_subplot(gs[0])
        ax_detail = self.figure.add_subplot(gs[1])

        # Label axes for event handling
        ax_overview.set_label('overview')
        ax_detail.set_label('detail')

        # ===== OVERVIEW PLOT (downsampled full recording) =====
        # Downsample for performance (max 2000 points)
        downsample_factor = max(1, num_samples_total // 2000)
        if full_data.ndim == 1 or full_data.shape[1] == 1:
            overview_data = full_data[::downsample_factor].flatten()
        else:
            # Show mean of selected channels for overview
            overview_data = np.mean(full_data[::downsample_factor, :min(10, full_data.shape[1])], axis=1)

        overview_time = np.arange(len(overview_data)) * downsample_factor / self.sampling_rate

        ax_overview.plot(overview_time, overview_data, 'k-', linewidth=0.5, alpha=0.6)
        ax_overview.set_ylabel('Overview', fontsize=9)
        ax_overview.set_xlim(0, num_samples_total / self.sampling_rate)
        ax_overview.tick_params(labelsize=8)
        ax_overview.grid(True, alpha=0.2)

        # Highlight current viewing window
        start_time = self.time_start / self.sampling_rate
        end_time = self.time_end / self.sampling_rate
        ax_overview.axvspan(start_time, end_time, alpha=0.3, color='blue', label='Current View')
        ax_overview.set_title('Full Recording (click to navigate)', fontsize=9, pad=3)

        # ===== DETAIL PLOT (current window) =====
        start_idx = int(self.time_start)
        end_idx = int(self.time_end)
        end_idx = min(end_idx, num_samples_total)
        start_idx = max(0, min(start_idx, end_idx - 1))

        data = full_data[start_idx:end_idx]
        if full_data.ndim > 1:
            data = data[:, channels_to_plot] if len(data) > 0 else np.array([])

        if len(data) == 0:
            return

        time_axis = (np.arange(len(data)) + start_idx) / self.sampling_rate

        if single_channel_mode:
            # Single channel view
            channel_data = data if data.ndim == 1 else data[:, 0]
            channel_data = channel_data * gain
            ax_detail.plot(time_axis, channel_data, 'k-', linewidth=0.7)
            ax_detail.set_xlabel('Time (s)')
            ax_detail.set_ylabel(f'Amplitude (Ch {channels_to_plot[0]+1})')
            ax_detail.set_title(f'Channel {channels_to_plot[0]+1} - Gain: {gain:.2f}x')
            ax_detail.grid(True, alpha=0.3)
        else:
            # Multi-channel stacked view (EEG style)
            data_range = np.ptp(data)
            if data_range == 0:
                data_range = 1.0

            channel_separation = data_range * separation_factor * gain
            colors = plt.cm.tab20(np.linspace(0, 1, len(channels_to_plot)))

            for i, ch_idx in enumerate(channels_to_plot):
                offset = i * channel_separation
                if data.ndim == 1:
                    channel_data = data * gain + offset
                else:
                    channel_data = data[:, i] * gain + offset

                ax_detail.plot(time_axis, channel_data, '-',
                              linewidth=0.5, color=colors[i], alpha=0.8)

                # Add channel label
                y_pos = offset
                label_x = time_axis[0] - (time_axis[-1] - time_axis[0]) * 0.012
                ax_detail.text(label_x, y_pos, f'Ch{ch_idx+1}',
                              verticalalignment='center', fontsize=8,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5))

            ax_detail.set_xlabel('Time (s)')
            ax_detail.set_ylabel('Channels')
            ax_detail.set_title(f'Detail View: {len(channels_to_plot)}/{full_data.shape[1]} channels '
                               f'(Gain: {gain:.2f}x, Sep: {separation_factor:.2f}x)', pad=5)
            ax_detail.set_xlim(time_axis[0], time_axis[-1])
            ax_detail.grid(True, alpha=0.2, axis='x')
            ax_detail.set_yticks([])

        self.figure.tight_layout()
        self.canvas.draw()


    def show_traces(self):
        """Display traces"""
        self.redraw()


    def show_spectrum(self):
        """Show power spectrum"""
        if len(self.meting.adc) == 0:
            self.message("No data loaded")
            return

        data = self.meting.adc if self.meting.adc.ndim == 1 else self.meting.adc[:, 0]

        # Compute power spectrum
        fs = 10000  # Sampling frequency
        freqs, psd = signal.welch(data, fs, nperseg=1024)

        # Plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.semilogy(freqs, psd)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Power Spectrum')
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


    def show_spectrogram(self):
        """Show spectrogram"""
        if len(self.meting.adc) == 0:
            return

        data = self.meting.adc if self.meting.adc.ndim == 1 else self.meting.adc[:, 0]

        fs = 10000
        freqs, times, Sxx = signal.spectrogram(data, fs, nperseg=256)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        pcm = ax.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Spectrogram')
        self.figure.colorbar(pcm, ax=ax, label='Power (dB)')

        self.figure.tight_layout()
        self.canvas.draw()


    # ========================================================================
    # ANALYSIS TOOLS
    # ========================================================================

    def do_psth(self):
        """Peri-stimulus time histogram"""
        self.message("PSTH Analysis\n(Implementation required)")


    def do_minis(self):
        """Miniature event analysis"""
        self.message("Mini Analysis\n(Implementation required)")


    def do_formula(self):
        """Formula evaluation"""
        self.message("Formula Evaluation\n(Implementation required)")


    # ========================================================================
    # FITTING AND CURVE ANALYSIS
    # ========================================================================

    def lsq_fit(self, func: Callable, x: np.ndarray, y: np.ndarray,
                p0: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Least squares fitting"""
        popt, pcov = curve_fit(func, x, y, p0=p0)
        return popt, pcov


    def nernst(self, ion_in: float, ion_out: float, valence: int = 1,
              temp: float = 310) -> float:
        """Calculate Nernst potential (mV)"""
        R = 8.314  # J/(mol·K)
        F = 96485  # C/mol
        return (R * temp / (valence * F)) * np.log(ion_out / ion_in) * 1000


    def ghk(self, pK: float, pNa: float, pCl: float,
           K_in: float, K_out: float,
           Na_in: float, Na_out: float,
           Cl_in: float, Cl_out: float,
           temp: float = 310) -> float:
        """Goldman-Hodgkin-Katz equation (mV)"""
        R = 8.314
        F = 96485

        numerator = pK * K_out + pNa * Na_out + pCl * Cl_in
        denominator = pK * K_in + pNa * Na_in + pCl * Cl_out

        return (R * temp / F) * np.log(numerator / denominator) * 1000


    # ========================================================================
    # ZOOM CONTROLS
    # ========================================================================

    def zoom_in(self):
        """Zoom in on data"""
        self.zoom.rate *= 1.5
        self.redraw()


    def zoom_out(self):
        """Zoom out on data"""
        self.zoom.rate /= 1.5
        self.redraw()


    def zoom_reset(self):
        """Reset zoom"""
        self.zoom = Zoom()
        self.redraw()


    # ========================================================================
    # PROTOCOL MANAGEMENT
    # ========================================================================

    def load_protocol(self):
        """Load protocol file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Protocol",
            self.file_settings.proto,
            "Protocol files (*.txt);;All files (*.*)"
        )

        if filename:
            self.statusBar().showMessage(f'Loaded protocol: {filename}')


    def save_protocol(self):
        """Save protocol file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Protocol",
            self.file_settings.proto,
            "Protocol files (*.txt);;All files (*.*)"
        )

        if filename:
            self.statusBar().showMessage(f'Saved protocol: {filename}')


    # ========================================================================
    # MEA FUNCTIONS
    # ========================================================================

    def mea_load_h5(self, filename: str):
        """Load MEA HDF5 file"""
        with h5py.File(filename, 'r') as f:
            if 'data' in f:
                data = f['data'][:]
                self.mea_data = MEAData(data=data)

                if 'sampling_rate' in f.attrs:
                    self.mea_data.sampling_rate = f.attrs['sampling_rate']


    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def linear_interpolate(self, x: np.ndarray, xp: np.ndarray,
                          yp: np.ndarray) -> np.ndarray:
        """Linear interpolation"""
        f = interp1d(xp, yp, kind='linear', bounds_error=False, fill_value='extrapolate')
        return f(x)


    def extreme(self, data: np.ndarray, mode: str = 'max') -> Tuple[int, float]:
        """Find extreme value"""
        if mode == 'max':
            idx = np.argmax(data)
        else:
            idx = np.argmin(data)
        return idx, data[idx]


    def local_variation(self, isis: np.ndarray) -> float:
        """Calculate local variation (LV) metric for spike train regularity"""
        if len(isis) < 2:
            return 0

        lv = 0
        for i in range(len(isis) - 1):
            lv += ((isis[i] - isis[i+1]) / (isis[i] + isis[i+1])) ** 2

        return (3.0 / (len(isis) - 1)) * lv


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = NeuronApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
