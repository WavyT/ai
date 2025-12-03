"""
Advanced EEG Preprocessing Tool with PyQtGraph
A high-performance graphical interface for EEG data visualization and preprocessing.

Features:
- High-performance visualization using PyQtGraph
- Real-time filtering (bandpass, highpass, lowpass, notch)
- Artifact detection and removal
- Spectral analysis (PSD, spectrogram)
- Channel operations (rereferencing, averaging)
- Interactive navigation (zoom, pan, channel selection)
- Export preprocessed data

Inspired by MATLAB Neuron.m visualization capabilities
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
        QListWidget, QListWidgetItem, QGroupBox, QSlider, QFileDialog,
        QMessageBox, QTabWidget, QTextEdit, QSplitter, QComboBox,
        QScrollArea, QGridLayout, QFrame, QProgressBar, QLineEdit
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
    from PyQt6.QtGui import QAction, QKeySequence, QFont
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
        QListWidget, QListWidgetItem, QGroupBox, QSlider, QFileDialog,
        QMessageBox, QTabWidget, QTextEdit, QSplitter, QComboBox,
        QScrollArea, QGridLayout, QFrame, QProgressBar, QLineEdit,
        QAction
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
    from PyQt5.QtGui import QKeySequence, QFont
    PYQT_VERSION = 5

try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget, ImageView
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    print("ERROR: PyQtGraph not installed. Please install it with: pip install pyqtgraph")

try:
    from scipy import signal
    from scipy.signal import butter, filtfilt, welch, spectrogram
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Some filtering features will be disabled.")

from eeg_loader import EEGLoader
import json
from datetime import datetime


class FilterWorker(QThread):
    """Worker thread for filtering operations to keep GUI responsive."""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, data, filter_type, params, fs):
        super().__init__()
        self.data = data
        self.filter_type = filter_type
        self.params = params
        self.fs = fs
    
    def run(self):
        try:
            filtered = self.apply_filter(self.data, self.filter_type, self.params, self.fs)
            self.finished.emit(filtered)
        except Exception as e:
            self.error.emit(str(e))
    
    def apply_filter(self, data, filter_type, params, fs):
        """Apply filter to data."""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for filtering")
        
        order = params.get('order', 4)
        filtered_data = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch].astype(np.float64)
            
            if filter_type == 'bandpass':
                low = params.get('low', 1.0)
                high = params.get('high', 100.0)
                nyq = fs / 2.0
                low_norm = low / nyq
                high_norm = high / nyq
                b, a = butter(order, [low_norm, high_norm], btype='band')
                filtered_data[:, ch] = filtfilt(b, a, channel_data)
            
            elif filter_type == 'highpass':
                cutoff = params.get('cutoff', 1.0)
                nyq = fs / 2.0
                cutoff_norm = cutoff / nyq
                b, a = butter(order, cutoff_norm, btype='high')
                filtered_data[:, ch] = filtfilt(b, a, channel_data)
            
            elif filter_type == 'lowpass':
                cutoff = params.get('cutoff', 100.0)
                nyq = fs / 2.0
                cutoff_norm = cutoff / nyq
                b, a = butter(order, cutoff_norm, btype='low')
                filtered_data[:, ch] = filtfilt(b, a, channel_data)
            
            elif filter_type == 'notch':
                freq = params.get('freq', 50.0)
                quality = params.get('quality', 30.0)
                w0 = freq / (fs / 2.0)
                b, a = signal.iirnotch(w0, quality)
                filtered_data[:, ch] = filtfilt(b, a, channel_data)
        
        return filtered_data


class AdvancedEEGGUI(QMainWindow):
    """Advanced EEG preprocessing GUI using PyQtGraph."""
    
    def __init__(self):
        super().__init__()
        
        if not HAS_PYQTGRAPH:
            QMessageBox.critical(
                None, "Error",
                "PyQtGraph is required but not installed.\n"
                "Please install it with: pip install pyqtgraph"
            )
            sys.exit(1)
        
        # Data storage
        self.loader: Optional[EEGLoader] = None
        self.raw_data: Optional[np.ndarray] = None
        self.processed_data: Optional[np.ndarray] = None
        self.current_data: Optional[np.ndarray] = None
        self.selected_channels: List[int] = []
        self.start_sample: int = 0
        self.end_sample: int = 10000
        self.sampling_rate: float = 200.0
        
        # Auto-loading tracking
        self.loaded_start_sample: int = 0  # Start of currently loaded data (with buffer)
        self.loaded_end_sample: int = 0    # End of currently loaded data (with buffer)
        self.load_buffer_ratio: float = 0.5  # Load 50% extra data on each side as buffer
        self.auto_load_enabled: bool = True  # Enable/disable auto-loading
        self.range_update_timer: Optional[QTimer] = None  # Debounce timer for range changes
        self.updating_view: bool = False  # Flag to prevent recursive updates
        
        # Filter chain
        self.filter_chain: List[Dict[str, Any]] = []
        
        # Scale controls
        self.x_scale_factor: float = 1.0  # X-axis (time) scale factor
        self.channel_y_scales: Dict[int, float] = {}  # Per-channel Y-scale factors
        self.base_y_scale: float = 1.0  # Base Y-scale that all channels are synced to
        self.channel_spacing: float = 1000.0  # Spacing between channel traces
        
        # Overview/minimap
        self.overview_channel: Optional[int] = None  # Channel to show in overview
        self.overview_data: Optional[np.ndarray] = None  # Downsampled overview data
        self.overview_loaded: bool = False  # Whether overview has been loaded
        
        # Initialize UI
        self.init_ui()
        
        # Try to auto-load continuous.dat
        self.auto_load_data()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Advanced EEG Preprocessing Tool (PyQtGraph)")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel: Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Visualization
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (1:3 ratio)
        splitter.setSizes([400, 1200])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_menu_bar(self):
        """Create menu bar with file and processing options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load EEG File...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)
        
        export_action = QAction('Export Processed Data...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        zoom_all_action = QAction('Zoom to Full Data', self)
        zoom_all_action.setShortcut('Ctrl+A')
        zoom_all_action.triggered.connect(self.zoom_to_full)
        view_menu.addAction(zoom_all_action)
        
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('Ctrl+R')
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
    
    def create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # File loading section
        file_group = self.create_file_group()
        scroll_layout.addWidget(file_group)
        
        # Channel selection section
        channel_group = self.create_channel_group()
        scroll_layout.addWidget(channel_group)
        
        # Time navigation section
        time_group = self.create_time_navigation_group()
        scroll_layout.addWidget(time_group)
        
        # Filtering section
        filter_group = self.create_filter_group()
        scroll_layout.addWidget(filter_group)
        
        # Processing section
        process_group = self.create_processing_group()
        scroll_layout.addWidget(process_group)
        
        # Export section
        export_group = self.create_export_group()
        scroll_layout.addWidget(export_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return panel
    
    def create_file_group(self) -> QGroupBox:
        """Create file loading group."""
        group = QGroupBox("File Loading")
        layout = QVBoxLayout()
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray;")
        layout.addWidget(self.file_label)
        
        load_btn = QPushButton("Load EEG File...")
        load_btn.clicked.connect(self.load_file)
        layout.addWidget(load_btn)
        
        # File info
        self.file_info_label = QLabel("")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.file_info_label)
        
        group.setLayout(layout)
        return group
    
    def create_channel_group(self) -> QGroupBox:
        """Create channel selection group."""
        group = QGroupBox("Channel Selection")
        layout = QVBoxLayout()
        
        # Channel list
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.channel_list.itemSelectionChanged.connect(self.on_channel_selection_changed)
        layout.addWidget(self.channel_list)
        
        # Channel buttons
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_channels)
        btn_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_channels)
        btn_layout.addWidget(deselect_all_btn)
        
        layout.addLayout(btn_layout)
        
        # Selected channels info
        self.selected_channels_label = QLabel("Selected: 0 channels")
        self.selected_channels_label.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.selected_channels_label)
        
        group.setLayout(layout)
        return group
    
    def create_time_navigation_group(self) -> QGroupBox:
        """Create time navigation group."""
        group = QGroupBox("Time Navigation")
        layout = QVBoxLayout()
        
        # Auto-load checkbox
        self.auto_load_checkbox = QCheckBox("Auto-load on pan/zoom")
        self.auto_load_checkbox.setChecked(True)
        self.auto_load_checkbox.stateChanged.connect(self.on_auto_load_changed)
        layout.addWidget(self.auto_load_checkbox)
        
        # Current view display (shows what's currently visible)
        self.view_range_label = QLabel("View: 0.00 - 0.00 s")
        self.view_range_label.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.view_range_label)
        
        # Time range inputs
        time_layout = QGridLayout()
        
        time_layout.addWidget(QLabel("Start Sample:"), 0, 0)
        self.start_sample_spin = QSpinBox()
        self.start_sample_spin.setMaximum(999999999)
        self.start_sample_spin.setValue(0)
        self.start_sample_spin.valueChanged.connect(self.on_time_range_changed)
        time_layout.addWidget(self.start_sample_spin, 0, 1)
        
        time_layout.addWidget(QLabel("End Sample:"), 1, 0)
        self.end_sample_spin = QSpinBox()
        self.end_sample_spin.setMaximum(999999999)
        self.end_sample_spin.setValue(10000)
        self.end_sample_spin.valueChanged.connect(self.on_time_range_changed)
        time_layout.addWidget(self.end_sample_spin, 1, 1)
        
        layout.addLayout(time_layout)
        
        # Time duration display
        self.duration_label = QLabel("Duration: 0.00 s")
        layout.addWidget(self.duration_label)
        
        # Navigation buttons - improved layout
        nav_layout = QGridLayout()
        
        btn_start = QPushButton("⏮ Start")
        btn_start.clicked.connect(self.go_to_start)
        nav_layout.addWidget(btn_start, 0, 0)
        
        btn_prev_10k = QPushButton("◀◀ -10k")
        btn_prev_10k.clicked.connect(lambda: self.navigate_time(-10000))
        nav_layout.addWidget(btn_prev_10k, 0, 1)
        
        btn_prev_5k = QPushButton("◀ -5k")
        btn_prev_5k.clicked.connect(lambda: self.navigate_time(-5000))
        nav_layout.addWidget(btn_prev_5k, 0, 2)
        
        btn_next_5k = QPushButton("+5k ▶")
        btn_next_5k.clicked.connect(lambda: self.navigate_time(5000))
        nav_layout.addWidget(btn_next_5k, 0, 3)
        
        btn_next_10k = QPushButton("+10k ▶▶")
        btn_next_10k.clicked.connect(lambda: self.navigate_time(10000))
        nav_layout.addWidget(btn_next_10k, 0, 4)
        
        btn_end = QPushButton("End ⏭")
        btn_end.clicked.connect(self.go_to_end)
        nav_layout.addWidget(btn_end, 0, 5)
        
        layout.addLayout(nav_layout)
        
        # Load button
        load_update_btn = QPushButton("Load & Update View")
        load_update_btn.clicked.connect(self.load_and_update_view)
        layout.addWidget(load_update_btn)
        
        group.setLayout(layout)
        return group
    
    def create_filter_group(self) -> QGroupBox:
        """Create filtering controls group."""
        group = QGroupBox("Filtering")
        layout = QVBoxLayout()
        
        if not HAS_SCIPY:
            no_scipy_label = QLabel("⚠ scipy not installed.\nFiltering disabled.")
            no_scipy_label.setStyleSheet("color: red;")
            layout.addWidget(no_scipy_label)
            group.setLayout(layout)
            return group
        
        # Filter type selection
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Type:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(['bandpass', 'highpass', 'lowpass', 'notch'])
        layout.addLayout(filter_type_layout)
        layout.addWidget(self.filter_type_combo)
        
        # Filter parameters
        param_layout = QGridLayout()
        
        # Low frequency (for bandpass)
        param_layout.addWidget(QLabel("Low (Hz):"), 0, 0)
        self.filter_low_spin = QDoubleSpinBox()
        self.filter_low_spin.setRange(0.1, 1000.0)
        self.filter_low_spin.setValue(1.0)
        self.filter_low_spin.setDecimals(2)
        param_layout.addWidget(self.filter_low_spin, 0, 1)
        
        # High frequency (for bandpass)
        param_layout.addWidget(QLabel("High (Hz):"), 1, 0)
        self.filter_high_spin = QDoubleSpinBox()
        self.filter_high_spin.setRange(0.1, 1000.0)
        self.filter_high_spin.setValue(100.0)
        self.filter_high_spin.setDecimals(2)
        param_layout.addWidget(self.filter_high_spin, 1, 1)
        
        # Notch frequency
        param_layout.addWidget(QLabel("Notch Freq (Hz):"), 2, 0)
        self.filter_notch_spin = QDoubleSpinBox()
        self.filter_notch_spin.setRange(1.0, 200.0)
        self.filter_notch_spin.setValue(50.0)
        self.filter_notch_spin.setDecimals(2)
        param_layout.addWidget(self.filter_notch_spin, 2, 1)
        
        # Filter order
        param_layout.addWidget(QLabel("Order:"), 3, 0)
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 10)
        self.filter_order_spin.setValue(4)
        param_layout.addWidget(self.filter_order_spin, 3, 1)
        
        layout.addLayout(param_layout)
        
        # Filter buttons
        apply_filter_btn = QPushButton("Apply Filter")
        apply_filter_btn.clicked.connect(self.apply_filter)
        layout.addWidget(apply_filter_btn)
        
        clear_filter_btn = QPushButton("Clear Filter")
        clear_filter_btn.clicked.connect(self.clear_filter)
        layout.addWidget(clear_filter_btn)
        
        # Filter chain display
        self.filter_chain_label = QLabel("Filters: None")
        self.filter_chain_label.setWordWrap(True)
        self.filter_chain_label.setStyleSheet("font-size: 9pt; color: gray;")
        layout.addWidget(self.filter_chain_label)
        
        group.setLayout(layout)
        return group
    
    def create_processing_group(self) -> QGroupBox:
        """Create signal processing controls group."""
        group = QGroupBox("Signal Processing")
        layout = QVBoxLayout()
        
        # DC removal
        self.dc_remove_check = QCheckBox("Remove DC offset")
        self.dc_remove_check.setChecked(True)
        self.dc_remove_check.stateChanged.connect(self.on_processing_changed)
        layout.addWidget(self.dc_remove_check)
        
        # Normalization
        self.normalize_check = QCheckBox("Normalize (z-score)")
        self.normalize_check.setChecked(False)  # Default to off - show raw data first
        self.normalize_check.stateChanged.connect(self.on_processing_changed)
        layout.addWidget(self.normalize_check)
        
        # Rereferencing
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Rereference:"))
        self.reref_combo = QComboBox()
        self.reref_combo.addItems(['None', 'Average', 'Common Average', 'Bipolar'])
        self.reref_combo.currentTextChanged.connect(self.on_processing_changed)
        ref_layout.addWidget(self.reref_combo)
        layout.addLayout(ref_layout)
        
        # Apply processing button
        apply_processing_btn = QPushButton("Apply Processing")
        apply_processing_btn.clicked.connect(self.apply_processing)
        layout.addWidget(apply_processing_btn)
        
        group.setLayout(layout)
        return group
    
    def create_export_group(self) -> QGroupBox:
        """Create export controls group."""
        group = QGroupBox("Export")
        layout = QVBoxLayout()
        
        export_btn = QPushButton("Export Processed Data...")
        export_btn.clicked.connect(self.export_data)
        layout.addWidget(export_btn)
        
        export_metadata_btn = QPushButton("Export Metadata...")
        export_metadata_btn.clicked.connect(self.export_metadata)
        layout.addWidget(export_metadata_btn)
        
        group.setLayout(layout)
        return group
    
    def create_right_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget for different views
        self.view_tabs = QTabWidget()
        
        # Time series view
        self.time_series_view = self.create_time_series_view()
        self.view_tabs.addTab(self.time_series_view, "Time Series")
        
        # Spectrogram view
        if HAS_SCIPY:
            self.spectrogram_view = self.create_spectrogram_view()
            self.view_tabs.addTab(self.spectrogram_view, "Spectrogram")
        
        # PSD view
        if HAS_SCIPY:
            self.psd_view = self.create_psd_view()
            self.view_tabs.addTab(self.psd_view, "Power Spectral Density")
        
        layout.addWidget(self.view_tabs)
        
        # Info label
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.info_label)
        
        return panel
    
    def create_time_series_view(self) -> QWidget:
        """Create time series visualization widget with overview and scale controls."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scale controls toolbar
        scale_toolbar = self.create_scale_controls()
        main_layout.addWidget(scale_toolbar)
        
        # Main plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        
        # Configure ViewBox for proper zoom/pan behavior on data axes
        vb = self.plot_widget.getViewBox()
        
        # Set mouse mode to PanMode - this allows:
        # - Left-click drag: pan the view
        # - Scroll wheel: zoom in/out on data axes (centered at mouse position)
        # This ensures zoom/pan works on data coordinates, not image coordinates
        vb.setMouseMode(vb.PanMode)
        
        # Enable zooming with scroll wheel on both axes
        vb.setMouseEnabled(x=True, y=True)
        
        # Disable auto-range so zoom/pan works correctly
        vb.enableAutoRange(axis='x', enable=False)
        vb.enableAutoRange(axis='y', enable=False)
        
        # Connect to range change signal to enable auto-loading when panning
        vb.sigRangeChanged.connect(self.on_view_range_changed)
        
        # Don't clip to view - show all data
        self.plot_widget.setClipToView(False)
        
        # Create debounce timer for range changes (wait 200ms after user stops panning)
        self.range_update_timer = QTimer()
        self.range_update_timer.setSingleShot(True)
        self.range_update_timer.timeout.connect(self.check_and_load_new_data)
        
        main_layout.addWidget(self.plot_widget, stretch=1)
        
        # Overview/minimap widget
        overview_widget = self.create_overview_widget()
        main_layout.addWidget(overview_widget)
        
        return widget
    
    def create_scale_controls(self) -> QWidget:
        """Create scale control toolbar."""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # X-scale controls
        layout.addWidget(QLabel("X-Scale:"))
        x_dec_btn = QPushButton("−")
        x_dec_btn.setToolTip("Decrease time scale (zoom out)")
        x_dec_btn.clicked.connect(self.decrease_x_scale)
        layout.addWidget(x_dec_btn)
        
        self.x_scale_label = QLabel("1.0x")
        self.x_scale_label.setMinimumWidth(50)
        layout.addWidget(self.x_scale_label)
        
        x_inc_btn = QPushButton("+")
        x_inc_btn.setToolTip("Increase time scale (zoom in)")
        x_inc_btn.clicked.connect(self.increase_x_scale)
        layout.addWidget(x_inc_btn)
        
        layout.addWidget(QFrame())  # Spacer
        
        # Y-scale controls
        layout.addWidget(QLabel("Y-Scale:"))
        y_dec_btn = QPushButton("−")
        y_dec_btn.setToolTip("Decrease amplitude scale")
        y_dec_btn.clicked.connect(self.decrease_y_scale)
        layout.addWidget(y_dec_btn)
        
        self.y_scale_label = QLabel("1.0x")
        self.y_scale_label.setMinimumWidth(50)
        layout.addWidget(self.y_scale_label)
        
        y_inc_btn = QPushButton("+")
        y_inc_btn.setToolTip("Increase amplitude scale")
        y_inc_btn.clicked.connect(self.increase_y_scale)
        layout.addWidget(y_inc_btn)
        
        layout.addStretch()
        
        return toolbar
    
    def create_overview_widget(self) -> QWidget:
        """Create overview/minimap widget showing full trace."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Overview header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Overview:"))
        
        self.overview_channel_combo = QComboBox()
        self.overview_channel_combo.currentIndexChanged.connect(self.on_overview_channel_changed)
        header_layout.addWidget(self.overview_channel_combo)
        header_layout.addWidget(QLabel("(Click/drag to select region)"))
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Overview plot widget
        self.overview_widget = pg.PlotWidget()
        self.overview_widget.setLabel('bottom', 'Time (s)')
        self.overview_widget.showGrid(x=True, y=True, alpha=0.3)
        self.overview_widget.setFixedHeight(150)  # Fixed height for overview
        self.overview_widget.setMouseEnabled(x=True, y=False)  # Only horizontal pan
        
        # Connect mouse events for region selection
        self.overview_widget.scene().sigMouseClicked.connect(self.on_overview_click)
        self.overview_widget.scene().sigMouseMoved.connect(self.on_overview_mouse_move)
        
        # View range indicator will be added when data is loaded
        self.overview_range_rect = None
        
        layout.addWidget(self.overview_widget)
        
        return container
    
    def create_spectrogram_view(self) -> QWidget:
        """Create spectrogram visualization widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # PyQtGraph ImageView for spectrogram
        self.spectrogram_viewer = ImageView()
        layout.addWidget(self.spectrogram_viewer)
        
        return widget
    
    def create_psd_view(self) -> QWidget:
        """Create power spectral density visualization widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # PyQtGraph plot widget for PSD
        self.psd_widget = pg.PlotWidget()
        self.psd_widget.setLabel('left', 'Power (dB)')
        self.psd_widget.setLabel('bottom', 'Frequency (Hz)')
        self.psd_widget.showGrid(x=True, y=True, alpha=0.3)
        self.psd_widget.addLegend()
        self.psd_widget.setLogMode(x=False, y=True)
        
        layout.addWidget(self.psd_widget)
        
        return widget
    
    def auto_load_data(self):
        """Try to automatically load continuous.dat if it exists."""
        dat_file = Path("continuous.dat")
        if dat_file.exists():
            try:
                self.load_file_path(str(dat_file))
            except Exception as e:
                QMessageBox.warning(self, "Auto-load Warning", f"Could not auto-load file: {e}")
    
    def load_file(self):
        """Open file dialog to load EEG file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select EEG Data File", "", "DAT files (*.dat);;All files (*.*)"
        )
        if filepath:
            self.load_file_path(filepath)
    
    def load_file_path(self, filepath: str):
        """Load EEG file from path."""
        try:
            self.statusBar().showMessage("Loading file...")
            QApplication.processEvents()
            
            # Initialize loader
            self.loader = EEGLoader(filepath, num_channels=72)
            
            # Update sampling rate
            if self.loader.timestamps is not None and len(self.loader.timestamps) > 1:
                self.sampling_rate = 1.0 / np.mean(np.diff(self.loader.timestamps))
            else:
                self.sampling_rate = 200.0
            
            # Update UI
            self.file_label.setText(f"Loaded: {Path(filepath).name}")
            self.file_label.setStyleSheet("color: green; font-weight: bold;")
            
            # File info
            duration = self.loader.num_samples_per_channel / self.sampling_rate
            info_text = (
                f"Channels: {self.loader.num_channels}\n"
                f"Samples: {self.loader.num_samples_per_channel:,}\n"
                f"Sampling rate: {self.sampling_rate:.2f} Hz\n"
                f"Duration: {duration/60:.2f} min"
            )
            self.file_info_label.setText(info_text)
            
            # Populate channel list
            self.channel_list.clear()
            for i in range(self.loader.num_channels):
                item = QListWidgetItem(f"Channel {i:3d}")
                self.channel_list.addItem(item)
            
            # Populate overview channel combo if it exists
            if hasattr(self, 'overview_channel_combo'):
                self.overview_channel_combo.clear()
                for i in range(self.loader.num_channels):
                    self.overview_channel_combo.addItem(f"Channel {i:3d}")
                if self.loader.num_channels > 0:
                    self.overview_channel = 0
                    self.overview_channel_combo.setCurrentIndex(0)
            
            # Set default time window
            max_samples = self.loader.num_samples_per_channel
            self.end_sample = min(10000, max_samples)
            self.start_sample_spin.setMaximum(max_samples)
            self.end_sample_spin.setMaximum(max_samples)
            self.start_sample_spin.setValue(0)
            self.end_sample_spin.setValue(self.end_sample)
            
            # Initialize loaded range (will be set properly when data is first loaded)
            self.loaded_start_sample = 0
            self.loaded_end_sample = 0
            
            self.statusBar().showMessage(f"Loaded {self.loader.num_channels} channels successfully")
            
            QMessageBox.information(
                self, "File Loaded",
                f"Loaded {self.loader.num_channels} channels\n"
                f"Total samples: {max_samples:,}\n"
                f"Sampling rate: {self.sampling_rate:.2f} Hz"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("Error loading file")
    
    def on_channel_selection_changed(self):
        """Handle channel selection changes."""
        selected_items = self.channel_list.selectedItems()
        self.selected_channels = [
            self.channel_list.row(item) for item in selected_items
        ]
        self.selected_channels_label.setText(
            f"Selected: {len(self.selected_channels)} channels"
        )
        if self.selected_channels:
            # Initialize Y-scales for new channels
            for ch_idx in self.selected_channels:
                if ch_idx not in self.channel_y_scales:
                    self.channel_y_scales[ch_idx] = self.base_y_scale
            
            # Load overview if not loaded and we have a selected channel for it
            if self.overview_channel is None and len(self.selected_channels) > 0:
                self.overview_channel = self.selected_channels[0]
                if hasattr(self, 'overview_channel_combo'):
                    # Find index in combo box
                    for i in range(self.overview_channel_combo.count()):
                        if f"Channel {self.overview_channel:3d}" in self.overview_channel_combo.itemText(i):
                            self.overview_channel_combo.setCurrentIndex(i)
                            break
                    self.load_overview_data()
            
            self.load_and_update_view()
    
    def select_all_channels(self):
        """Select all channels."""
        self.channel_list.selectAll()
        self.on_channel_selection_changed()
    
    def deselect_all_channels(self):
        """Deselect all channels."""
        self.channel_list.clearSelection()
        self.on_channel_selection_changed()
    
    def on_time_range_changed(self):
        """Handle time range spin box changes."""
        self.start_sample = self.start_sample_spin.value()
        self.end_sample = self.end_sample_spin.value()
        
        if self.end_sample <= self.start_sample:
            self.end_sample = self.start_sample + 1
            self.end_sample_spin.setValue(self.end_sample)
        
        duration = (self.end_sample - self.start_sample) / self.sampling_rate
        self.duration_label.setText(f"Duration: {duration:.2f} s")
    
    def on_auto_load_changed(self, state):
        """Handle auto-load checkbox state change."""
        self.auto_load_enabled = (state == Qt.CheckState.Checked.value)
        if hasattr(self, 'auto_load_checkbox'):
            self.auto_load_enabled = self.auto_load_checkbox.isChecked()
    
    def on_view_range_changed(self):
        """Handle ViewBox range changes (when user pans/zooms)."""
        if not self.auto_load_enabled or self.updating_view:
            return
        
        # Debounce: restart timer every time range changes
        # This prevents loading while user is actively panning
        if self.range_update_timer:
            self.range_update_timer.stop()
            self.range_update_timer.start(300)  # Wait 300ms after user stops panning
        
        # Update overview range indicator immediately for responsive feedback
        if hasattr(self, 'overview_widget') and self.overview_loaded:
            self.update_overview_range_indicator()
    
    def check_and_load_new_data(self):
        """Check if visible range is outside loaded data and load if needed."""
        if not self.auto_load_enabled or self.loader is None or not self.selected_channels:
            return
        
        try:
            vb = self.plot_widget.getViewBox()
            if vb is None:
                return
            
            # Get current visible X range (in seconds, since we're plotting in seconds)
            visible_range = vb.viewRange()[0]  # [x_min, x_max] in seconds
            visible_start_sec = visible_range[0]
            visible_end_sec = visible_range[1]
            
            # Convert from seconds to sample indices
            visible_start = int(visible_start_sec * self.sampling_rate)
            visible_end = int(visible_end_sec * self.sampling_rate)
            
            # Clamp to valid range
            max_samples = self.loader.num_samples_per_channel
            visible_start = max(0, min(visible_start, max_samples))
            visible_end = max(0, min(visible_end, max_samples))
            
            if visible_end <= visible_start:
                return
            
            # Enforce minimum window size to prevent loading single samples
            visible_range_size = visible_end - visible_start
            min_window_size = 100  # Minimum samples to display
            
            if visible_range_size < min_window_size:
                # Expand to minimum size, centered on current view
                center = (visible_start + visible_end) / 2
                visible_start = max(0, int(center - min_window_size / 2))
                visible_end = min(max_samples, visible_start + min_window_size)
                visible_range_size = visible_end - visible_start
            
            # Calculate buffer size (50% of visible range on each side)
            buffer_size = int(visible_range_size * self.load_buffer_ratio)
            
            # Calculate desired load range (with buffer)
            desired_start = max(0, visible_start - buffer_size)
            desired_end = min(max_samples, visible_end + buffer_size)
            
            # Check if we need to load new data
            need_load = False
            margin = max(visible_range_size * 0.1, 1000)  # 10% margin, but at least 1000 samples
            
            # Also reload if resolution needs change significantly (when zooming in/out)
            # If we're zoomed in very close, we need full resolution data
            current_loaded_range = self.loaded_end_sample - self.loaded_start_sample
            resolution_needs_reload = False
            
            # Check if we need to reload for better resolution
            visible_range_seconds = visible_range_size / self.sampling_rate
            
            if visible_range_seconds < 2.0 and current_loaded_range > visible_range_size * 5:
                # Zoomed in close (< 2 seconds) but have too much buffered data - reload for full resolution
                resolution_needs_reload = True
            elif visible_range_seconds > 60.0 and current_loaded_range < visible_range_size * 2:
                # Zoomed out far (> 60 seconds) and need more data
                resolution_needs_reload = True
            
            if (desired_start < self.loaded_start_sample - margin or 
                desired_end > self.loaded_end_sample + margin or
                resolution_needs_reload):
                need_load = True
            
            if need_load:
                print(f"Auto-loading: visible=[{visible_start:,}, {visible_end:,}], "
                      f"loading=[{desired_start:,}, {desired_end:,}]")
                
                # Update loaded range tracking
                self.loaded_start_sample = desired_start
                self.loaded_end_sample = desired_end
                
                # Update navigation controls to match visible range (but don't trigger reload)
                # The view will update automatically when data is loaded
                self.start_sample = visible_start
                self.end_sample = visible_end
                
                # Load data with buffer
                self.load_and_update_view_range(desired_start, desired_end, visible_start, visible_end)
                
                # Update navigation controls after loading to reflect actual loaded range
                self.start_sample_spin.blockSignals(True)
                self.end_sample_spin.blockSignals(True)
                self.start_sample_spin.setValue(visible_start)
                self.end_sample_spin.setValue(visible_end)
                self.start_sample_spin.blockSignals(False)
                self.end_sample_spin.blockSignals(False)
        
        except Exception as e:
            print(f"Error in check_and_load_new_data: {e}")
            import traceback
            traceback.print_exc()
    
    def load_and_update_view_range(self, load_start: int, load_end: int, 
                                   view_start: int, view_end: int):
        """Load data in range [load_start, load_end) and display [view_start, view_end)."""
        if self.loader is None or not self.selected_channels:
            return
        
        try:
            self.statusBar().showMessage("Auto-loading data...")
            QApplication.processEvents()
            
            # Load raw data with buffer
            self.raw_data = self.loader.load_channels(
                self.selected_channels,
                start_sample=load_start,
                end_sample=load_end,
                dtype=np.float32
            )
            
            # Update loaded range tracking
            self.loaded_start_sample = load_start
            self.loaded_end_sample = load_end
            
            # Apply processing
            self.current_data = self.raw_data.copy()
            self.apply_processing(internal=True)
            
            # Update plot with full loaded data, but set view to visible range
            self.updating_view = True  # Prevent recursive updates
            self.update_time_series_view()
            
            # Set view to show the desired visible range (convert samples to seconds)
            vb = self.plot_widget.getViewBox()
            if vb:
                vb.blockSignals(True)  # Temporarily block signals
                view_start_sec = view_start / self.sampling_rate
                view_end_sec = view_end / self.sampling_rate
                vb.setXRange(view_start_sec, view_end_sec, padding=0)
                vb.blockSignals(False)
            
            self.updating_view = False
            
            # Update view range label (convert to seconds)
            view_start_sec = view_start / self.sampling_rate
            view_end_sec = view_end / self.sampling_rate
            load_start_sec = load_start / self.sampling_rate
            load_end_sec = load_end / self.sampling_rate
            self.view_range_label.setText(
                f"View: {view_start_sec:.2f} - {view_end_sec:.2f} s "
                f"(Loaded: {load_start_sec:.2f} - {load_end_sec:.2f} s)"
            )
            
            # Update overview range indicator
            if hasattr(self, 'overview_widget') and self.overview_loaded:
                QApplication.processEvents()  # Ensure view is updated first
                self.update_overview_range_indicator()
            
            self.statusBar().showMessage("Data auto-loaded")
        
        except Exception as e:
            print(f"Error in load_and_update_view_range: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("Error auto-loading data")
    
    def go_to_start(self):
        """Navigate to the start of the data."""
        if self.loader is None:
            return
        
        window_size = self.end_sample - self.start_sample
        self.start_sample = 0
        self.end_sample = min(window_size, self.loader.num_samples_per_channel)
        self.start_sample_spin.setValue(self.start_sample)
        self.end_sample_spin.setValue(self.end_sample)
        self.load_and_update_view()
    
    def go_to_end(self):
        """Navigate to the end of the data."""
        if self.loader is None:
            return
        
        window_size = self.end_sample - self.start_sample
        max_samples = self.loader.num_samples_per_channel
        self.end_sample = max_samples
        self.start_sample = max(0, max_samples - window_size)
        self.start_sample_spin.setValue(self.start_sample)
        self.end_sample_spin.setValue(self.end_sample)
        self.load_and_update_view()
    
    def navigate_time(self, offset: int):
        """Navigate in time by offset samples."""
        if self.loader is None:
            return
        
        window_size = self.end_sample - self.start_sample
        new_start = max(0, self.start_sample + offset)
        new_end = min(self.loader.num_samples_per_channel, new_start + window_size)
        
        if new_end - new_start < 100:  # Minimum window size
            return
        
        self.start_sample = new_start
        self.end_sample = new_end
        self.start_sample_spin.setValue(self.start_sample)
        self.end_sample_spin.setValue(self.end_sample)
        self.load_and_update_view()
    
    def load_and_update_view(self):
        """Load data for current selection and update views."""
        if self.loader is None or not self.selected_channels:
            return
        
        try:
            self.statusBar().showMessage("Loading data...")
            QApplication.processEvents()
            
            # Calculate buffer for auto-loading
            window_size = self.end_sample - self.start_sample
            # Enforce minimum window size
            min_window_size = 100
            if window_size < min_window_size:
                center = (self.start_sample + self.end_sample) / 2
                self.start_sample = max(0, int(center - min_window_size / 2))
                self.end_sample = min(self.loader.num_samples_per_channel, self.start_sample + min_window_size)
                window_size = self.end_sample - self.start_sample
                self.start_sample_spin.setValue(self.start_sample)
                self.end_sample_spin.setValue(self.end_sample)
            
            buffer_size = int(window_size * self.load_buffer_ratio)
            load_start = max(0, self.start_sample - buffer_size)
            load_end = min(self.loader.num_samples_per_channel, self.end_sample + buffer_size)
            
            # Load raw data with buffer
            self.raw_data = self.loader.load_channels(
                self.selected_channels,
                start_sample=load_start,
                end_sample=load_end,
                dtype=np.float32
            )
            
            # Update loaded range tracking
            self.loaded_start_sample = load_start
            self.loaded_end_sample = load_end
            
            # Debug: Check data shape and values
            print(f"Loaded raw data shape: {self.raw_data.shape}")
            print(f"Raw data range: [{self.raw_data.min():.2f}, {self.raw_data.max():.2f}]")
            print(f"Loaded range: [{load_start:,}, {load_end:,}], Display range: [{self.start_sample:,}, {self.end_sample:,}]")
            
            # Apply processing
            self.current_data = self.raw_data.copy()
            self.apply_processing(internal=True)
            
            # Debug: Check processed data
            print(f"Processed data shape: {self.current_data.shape}")
            print(f"Processed data range: [{self.current_data.min():.2f}, {self.current_data.max():.2f}]")
            
            # Update views
            self.updating_view = True  # Prevent recursive range change triggers
            self.update_time_series_view()
            if HAS_SCIPY:
                self.update_spectrogram_view()
                self.update_psd_view()
            self.updating_view = False
            
            # Update view range label (convert to seconds)
            view_start_sec = self.start_sample / self.sampling_rate
            view_end_sec = self.end_sample / self.sampling_rate
            load_start_sec = load_start / self.sampling_rate
            load_end_sec = load_end / self.sampling_rate
            self.view_range_label.setText(
                f"View: {view_start_sec:.2f} - {view_end_sec:.2f} s "
                f"(Loaded: {load_start_sec:.2f} - {load_end_sec:.2f} s)"
            )
            
            # Update overview range indicator if overview exists
            if hasattr(self, 'overview_widget') and self.overview_loaded:
                QApplication.processEvents()  # Ensure view is updated first
                self.update_overview_range_indicator()
            
            # Update info
            info_text = (
                f"Channels: {len(self.selected_channels)} | "
                f"Samples: {self.end_sample - self.start_sample:,} | "
                f"Duration: {(self.end_sample - self.start_sample)/self.sampling_rate:.2f}s | "
                f"Sampling rate: {self.sampling_rate:.2f} Hz"
            )
            self.info_label.setText(info_text)
            
            self.statusBar().showMessage("Data loaded and displayed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load/update data: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("Error loading data")
    
    # ==================== Scale Control Methods ====================
    
    def increase_x_scale(self):
        """Increase X-axis (time) scale (zoom in)."""
        self.x_scale_factor *= 1.5
        self.x_scale_label.setText(f"{self.x_scale_factor:.2f}x")
        self.apply_x_scale()
    
    def decrease_x_scale(self):
        """Decrease X-axis (time) scale (zoom out)."""
        self.x_scale_factor /= 1.5
        if self.x_scale_factor < 0.1:
            self.x_scale_factor = 0.1
        self.x_scale_label.setText(f"{self.x_scale_factor:.2f}x")
        self.apply_x_scale()
    
    def apply_x_scale(self):
        """Apply X-scale factor to current view."""
        if self.loader is None:
            return
        
        vb = self.plot_widget.getViewBox()
        if vb is None:
            return
        
        # Get current center and range
        x_range = vb.viewRange()[0]
        current_center = (x_range[0] + x_range[1]) / 2
        current_width = x_range[1] - x_range[0]
        
        # Scale the width (current_width is in seconds)
        new_width = current_width / self.x_scale_factor
        
        # Enforce minimum window size (in seconds)
        min_width_sec = 0.05  # Minimum 50ms window
        if new_width < min_width_sec:
            new_width = min_width_sec
        
        max_seconds = self.loader.num_samples_per_channel / self.sampling_rate
        new_start_sec = max(0, current_center - new_width / 2)
        new_end_sec = min(max_seconds, current_center + new_width / 2)
        
        # Update stored values (convert back to samples)
        self.start_sample = int(new_start_sec * self.sampling_rate)
        self.end_sample = int(new_end_sec * self.sampling_rate)
        
        # Set new range - this will trigger auto-loading if needed
        self.updating_view = True
        vb.setXRange(new_start_sec, new_end_sec, padding=0)
        self.updating_view = False
        
        # Update spin boxes
        self.start_sample_spin.blockSignals(True)
        self.end_sample_spin.blockSignals(True)
        self.start_sample_spin.setValue(self.start_sample)
        self.end_sample_spin.setValue(self.end_sample)
        self.start_sample_spin.blockSignals(False)
        self.end_sample_spin.blockSignals(False)
        
        # Trigger reload with adaptive resolution
        if self.auto_load_enabled:
            # Trigger check after a short delay to allow view to update
            if self.range_update_timer:
                self.range_update_timer.stop()
                self.range_update_timer.start(100)  # Shorter delay for scale changes
    
    def increase_y_scale(self):
        """Increase Y-axis (amplitude) scale for all channels."""
        self.base_y_scale *= 1.5
        self.y_scale_label.setText(f"{self.base_y_scale:.2f}x")
        # Update all channel scales
        for ch_idx in self.selected_channels:
            if ch_idx not in self.channel_y_scales:
                self.channel_y_scales[ch_idx] = self.base_y_scale
            else:
                self.channel_y_scales[ch_idx] *= 1.5
        self.update_time_series_view()
    
    def decrease_y_scale(self):
        """Decrease Y-axis (amplitude) scale for all channels."""
        self.base_y_scale /= 1.5
        if self.base_y_scale < 0.1:
            self.base_y_scale = 0.1
        self.y_scale_label.setText(f"{self.base_y_scale:.2f}x")
        # Update all channel scales
        for ch_idx in self.selected_channels:
            if ch_idx not in self.channel_y_scales:
                self.channel_y_scales[ch_idx] = self.base_y_scale
            else:
                self.channel_y_scales[ch_idx] /= 1.5
                if self.channel_y_scales[ch_idx] < 0.1:
                    self.channel_y_scales[ch_idx] = 0.1
        self.update_time_series_view()
    
    # ==================== Overview Widget Methods ====================
    
    def on_overview_channel_changed(self, index: int):
        """Handle overview channel selection change."""
        if self.loader is None or index < 0:
            return
        
        channel_text = self.overview_channel_combo.itemText(index)
        try:
            ch_num = int(channel_text.split()[-1])
            self.overview_channel = ch_num
            self.load_overview_data()
        except (ValueError, IndexError):
            pass
    
    def load_overview_data(self):
        """Load downsampled overview data for selected channel."""
        if self.loader is None or self.overview_channel is None:
            return
        
        try:
            max_samples = self.loader.num_samples_per_channel
            # Downsample to ~10000 points for overview
            overview_points = 10000
            downsample = max(1, max_samples // overview_points)
            
            # Load every Nth sample for overview
            sample_indices = np.arange(0, max_samples, downsample)
            
            # Load data in chunks to be memory efficient
            chunk_size = 100000
            overview_chunks = []
            
            for start_idx in range(0, max_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, max_samples)
                chunk = self.loader.load_channels(
                    [self.overview_channel],
                    start_sample=start_idx,
                    end_sample=end_idx,
                    dtype=np.float32
                )
                # Downsample chunk
                chunk_indices = sample_indices[(sample_indices >= start_idx) & (sample_indices < end_idx)]
                chunk_idx_local = chunk_indices - start_idx
                if len(chunk_idx_local) > 0:
                    overview_chunks.append(chunk[chunk_idx_local, 0])
            
            if overview_chunks:
                self.overview_data = np.concatenate(overview_chunks)
                self.overview_loaded = True
                self.update_overview_plot()
        except Exception as e:
            print(f"Error loading overview data: {e}")
            import traceback
            traceback.print_exc()
    
    def update_overview_plot(self):
        """Update the overview plot."""
        if self.overview_data is None or not self.overview_loaded:
            return
        
        self.overview_widget.clear()
        
        max_samples = self.loader.num_samples_per_channel
        overview_points = len(self.overview_data)
        # Convert to seconds for overview
        max_seconds = max_samples / self.sampling_rate
        time_axis = np.linspace(0, max_seconds, overview_points)
        
        # Plot overview trace
        pen = pg.mkPen('w', width=1)
        self.overview_widget.plot(time_axis, self.overview_data, pen=pen)
        
        # Add view range indicator
        self.update_overview_range_indicator()
        
        # Auto-range Y axis
        self.overview_widget.autoRange()
        self.overview_widget.setXRange(0, max_seconds)
    
    def update_overview_range_indicator(self):
        """Update the view range indicator rectangle in overview."""
        if self.overview_widget is None or self.loader is None:
            return
        
        max_samples = self.loader.num_samples_per_channel
        vb_overview = self.overview_widget.getViewBox()
        if vb_overview is None:
            return
        
        # Get current view range from main plot ViewBox (more accurate than start_sample/end_sample)
        # ViewBox range is in seconds, convert back to samples
        vb_main = self.plot_widget.getViewBox() if hasattr(self, 'plot_widget') else None
        if vb_main is not None:
            x_range = vb_main.viewRange()[0]  # [x_min, x_max] in seconds
            view_start_sec = x_range[0]
            view_end_sec = x_range[1]
            view_start = max(0, min(int(view_start_sec * self.sampling_rate), max_samples))
            view_end = max(0, min(int(view_end_sec * self.sampling_rate), max_samples))
        else:
            # Fallback to stored values
            view_start = max(0, min(self.start_sample, max_samples))
            view_end = max(0, min(self.end_sample, max_samples))
        
        view_width = view_end - view_start
        
        # Get current Y range for overview
        y_range = vb_overview.viewRange()[1]
        y_min, y_max = y_range[0], y_range[1]
        
        # Remove old rectangle if exists
        if self.overview_range_rect is not None:
            self.overview_widget.removeItem(self.overview_range_rect)
        
        # Convert view range from samples to seconds for overview
        view_start_sec = view_start / self.sampling_rate
        view_end_sec = view_end / self.sampling_rate
        
        # Create rectangle outline using plot lines
        rect_x = [view_start_sec, view_start_sec, view_end_sec, view_end_sec, view_start_sec]
        rect_y = [y_min, y_max, y_max, y_min, y_min]
        
        # Create rectangle item
        self.overview_range_rect = pg.PlotDataItem(
            rect_x, rect_y,
            pen=pg.mkPen('y', width=3),
            connect='all'
        )
        self.overview_range_rect.setZValue(10)
        self.overview_widget.addItem(self.overview_range_rect)
    
    def on_overview_click(self, event):
        """Handle mouse clicks on overview to navigate."""
        if self.overview_widget is None or self.loader is None:
            return
        
        # Get click position in data coordinates (in seconds)
        pos = self.overview_widget.plotItem.vb.mapSceneToView(event.scenePos())
        x_pos_sec = pos.x()
        
        # Convert from seconds to sample index
        max_samples = self.loader.num_samples_per_channel
        max_seconds = max_samples / self.sampling_rate
        x_pos_sec = max(0, min(x_pos_sec, max_seconds))
        x_pos = int(x_pos_sec * self.sampling_rate)
        
        # Navigate to clicked position
        window_size = self.end_sample - self.start_sample
        new_start = max(0, x_pos - window_size // 2)
        new_end = min(max_samples, new_start + window_size)
        
        if new_end - new_start >= 100:
            self.start_sample = new_start
            self.end_sample = new_end
            self.start_sample_spin.setValue(self.start_sample)
            self.end_sample_spin.setValue(self.end_sample)
            self.load_and_update_view()
    
    def on_overview_mouse_move(self, event):
        """Handle mouse movement on overview for dragging."""
        # Could implement drag-to-select region in future
        pass
    
    def update_time_series_view(self):
        """Update the time series plot."""
        if self.current_data is None:
            return
        
        if self.current_data.size == 0 or len(self.selected_channels) == 0:
            return
        
        self.plot_widget.clear()
        
        # Adaptive downsampling based on visible range and zoom level
        # Get current visible range from ViewBox (in samples)
        vb = self.plot_widget.getViewBox()
        visible_start_sample = self.loaded_start_sample
        visible_end_sample = self.loaded_end_sample
        
        if vb is not None and not self.updating_view:
            try:
                x_range = vb.viewRange()[0]  # [x_min, x_max] in seconds
                # Convert from seconds to sample indices
                visible_start_sample = max(self.loaded_start_sample, int(x_range[0] * self.sampling_rate))
                visible_end_sample = min(self.loaded_end_sample, int(x_range[1] * self.sampling_rate))
                visible_range_size = visible_end_sample - visible_start_sample
            except:
                visible_range_size = self.end_sample - self.start_sample
                visible_start_sample = self.start_sample
                visible_end_sample = self.end_sample
        else:
            visible_range_size = self.end_sample - self.start_sample
            visible_start_sample = self.start_sample
            visible_end_sample = self.end_sample
        
        # Adaptive resolution: always extract visible range from full-resolution data when zoomed in
        # Use adaptive downsampling only when zoomed out
        
        # Determine if we should use full resolution (zoomed in) or downsampled (zoomed out)
        # Use full resolution if visible range is less than 2 seconds worth of data
        # This ensures high precision when viewing seconds-level detail
        max_samples_for_full_res = int(2 * self.sampling_rate)  # 2 seconds (400 samples at 200 Hz)
        
        # Also check visible range in seconds for better threshold
        visible_range_seconds = visible_range_size / self.sampling_rate
        
        if visible_range_seconds < 2.0 and visible_range_size > 0:
            # Zoomed in: extract visible range at FULL resolution (no downsampling)
            # Add a small buffer (10% on each side) to ensure smooth panning
            buffer_samples = max(1, int(visible_range_size * 0.1))
            extract_start = max(0, visible_start_sample - buffer_samples - self.loaded_start_sample)
            extract_end = min(self.current_data.shape[0], 
                              visible_end_sample + buffer_samples - self.loaded_start_sample)
            
            if extract_end > extract_start and extract_end <= self.current_data.shape[0]:
                # Extract visible portion from full-resolution data
                plot_data = self.current_data[extract_start:extract_end].copy()
                downsample_factor = 1  # No downsampling - full resolution!
                time_samples = self.loaded_start_sample + extract_start + np.arange(plot_data.shape[0])
                print(f"Zoomed in: plotting visible range [{visible_start_sample:,}, {visible_end_sample:,}] "
                      f"({visible_range_seconds:.3f} s) at FULL resolution ({plot_data.shape[0]:,} points, no downsampling)")
            else:
                # Fallback: use all loaded data
                plot_data = self.current_data.copy()
                downsample_factor = 1
                time_samples = self.loaded_start_sample + np.arange(plot_data.shape[0])
                print(f"Zoomed in (fallback): using all loaded data ({plot_data.shape[0]:,} points)")
        else:
            # Zoomed out: use adaptive downsampling on all loaded data
            # Calculate max points based on visible range - be more generous with points
            if visible_range_seconds < 10.0:
                # Medium zoom: use ~3-4 points per pixel for good detail
                points_per_pixel = 3.5
                max_points = int(visible_range_size * points_per_pixel)
                max_points = max(10000, min(max_points, 500000))  # Allow up to 500k points
            elif visible_range_seconds < 60.0:
                # Zoomed out: use ~2 points per pixel
                points_per_pixel = 2.0
                max_points = int(visible_range_size * points_per_pixel)
                max_points = max(20000, min(max_points, 200000))
            else:
                # Very zoomed out: use fixed max for performance
                max_points = 100000  # Increased from 50k for better detail
            
            plot_data = self.current_data.copy()
            downsample_factor = 1
            
            if plot_data.shape[0] > max_points:
                downsample_factor = max(1, plot_data.shape[0] // max_points)
                plot_data = plot_data[::downsample_factor]
                print(f"Adaptive downsampling: visible_range={visible_range_size:,} samples ({visible_range_seconds:.2f} s), "
                      f"max_points={max_points:,}, downsample_factor={downsample_factor} "
                      f"(from {self.current_data.shape[0]:,} to {plot_data.shape[0]:,} samples)")
            else:
                print(f"No downsampling needed: visible_range={visible_range_size:,} samples ({visible_range_seconds:.2f} s), "
                      f"data_points={plot_data.shape[0]:,}, max_points={max_points:,}")
            
            time_samples = self.loaded_start_sample + np.arange(plot_data.shape[0]) * downsample_factor
        
        # Convert sample indices to time in seconds for X-axis
        time_axis = time_samples / self.sampling_rate
        
        # Calculate offset for stacking channels - use a fixed offset based on data std
        # This is more reliable than using data range
        if plot_data.size > 0:
            # Use standard deviation to determine reasonable offset
            # But handle the case where data is very small (e.g., after normalization)
            data_std = np.std(plot_data)
            data_range = np.max(plot_data) - np.min(plot_data)
            
            # Calculate channel spacing based on data characteristics
            if data_range < 10:
                self.channel_spacing = max(data_std * 6.0, 5.0)  # Larger spacing for normalized data
            else:
                self.channel_spacing = max(data_std * 4.0, data_range * 1.5)  # Spacing for raw data
            print(f"Using channel spacing: {self.channel_spacing:.2f} (data std: {data_std:.2f}, data range: {data_range:.2f})")
        else:
            self.channel_spacing = 1000.0
        
        # Plot each channel
        # Generate colors - use color names for maximum visibility on black background
        color_palette = ['w', 'r', 'g', 'b', 'c', 'm', 'y', '#FFA500', '#FF00FF', '#00FFFF']  # white, red, green, blue, cyan, magenta, yellow, orange, fuchsia, aqua
        
        try:
            plots_added = 0
            for i, ch_idx in enumerate(self.selected_channels):
                if i >= plot_data.shape[1]:
                    continue
                    
                channel_data = plot_data[:, i]
                
                # Ensure we have valid data
                if channel_data.size == 0:
                    continue
                
                # For stacking: center each channel around zero, then add offset
                # This ensures channels are stacked vertically without overlap
                channel_mean = channel_data.mean()
                channel_data_centered = channel_data - channel_mean
                
                # Apply Y-scale factor for this channel (individual per channel)
                y_scale = self.channel_y_scales.get(ch_idx, self.base_y_scale)
                scaled_data = channel_data_centered * y_scale
                
                # Apply offset for stacking (multiply by channel index)
                y_data = scaled_data + (i * self.channel_spacing)
                
                # Debug: print actual y-values being plotted
                if i == 0:  # Only print for first channel to avoid spam
                    print(f"Channel {ch_idx} y-data sample: min={y_data.min():.2f}, max={y_data.max():.2f}, mean={y_data.mean():.2f}")
                    print(f"  Channel data (before offset): min={channel_data_centered.min():.2f}, max={channel_data_centered.max():.2f}")
                
                # Cycle through color palette - use bright colors for visibility
                color = color_palette[i % len(color_palette)]
                # Use color names directly - PyQtGraph understands these better
                # Increase width significantly for visibility
                pen = pg.mkPen(color=color, width=3)  # Bright colors, thicker lines for visibility
                
                # Plot the data - time_axis is already in seconds
                # Convert to numpy arrays if needed
                time_array = np.asarray(time_axis, dtype=np.float64)
                y_array = np.asarray(y_data, dtype=np.float64)
                
                # Ensure arrays are 1D and contiguous
                time_array = np.ravel(time_array)
                y_array = np.ravel(y_array)
                
                # Plot the data - use PlotDataItem for better control
                plot_item = pg.PlotDataItem(
                    time_array, y_array,
                    pen=pen,
                    name=f'Ch {ch_idx}'
                )
                self.plot_widget.addItem(plot_item)
                plots_added += 1
                print(f"Plotted channel {ch_idx}: {len(y_array)} points, y-range: [{y_array.min():.2f}, {y_array.max():.2f}], first 5 y-values: {y_array[:5]}")
            
            print(f"Successfully added {plots_added} plots to display")
        
        except Exception as e:
            print(f"Error plotting channels: {e}")
            import traceback
            traceback.print_exc()
        
        # Update label to show time range in seconds
        loaded_start_sec = self.loaded_start_sample / self.sampling_rate
        loaded_end_sec = self.loaded_end_sample / self.sampling_rate
        self.plot_widget.setLabel('bottom', f'Time (s) (Loaded: {loaded_start_sec:.2f} - {loaded_end_sec:.2f} s)')
        
        # Explicitly set view range to ensure data is visible
        if plots_added > 0:
            # DISABLE auto-range first - this is critical!
            self.plot_widget.disableAutoRange()
            
            # Get all plot items to calculate overall range
            plot_items = self.plot_widget.listDataItems()
            if plot_items:
                all_x_min = float('inf')
                all_x_max = float('-inf')
                all_y_min = float('inf')
                all_y_max = float('-inf')
                
                for item in plot_items:
                    x_data, y_data = item.getData()
                    if x_data is not None and y_data is not None and len(x_data) > 0:
                        all_x_min = min(all_x_min, float(np.min(x_data)))
                        all_x_max = max(all_x_max, float(np.max(x_data)))
                        all_y_min = min(all_y_min, float(np.min(y_data)))
                        all_y_max = max(all_y_max, float(np.max(y_data)))
                
                if (all_x_min < float('inf') and all_y_min < float('inf')):
                    # Only set Y range automatically, preserve X range if user is panning
                    # Don't override X range if we're updating from auto-load (preserve user's view)
                    vb = self.plot_widget.getViewBox()
                    vb.setAutoVisible(x=False, y=False)
                    vb.disableAutoRange()
                    
                    # Only set Y range (amplitude)
                    y_range = all_y_max - all_y_min
                    y_padding = max(y_range * 0.1, 10.0) if y_range > 0 else 100.0
                    y_min = all_y_min - y_padding
                    y_max = all_y_max + y_padding
                    
                    vb.setYRange(y_min, y_max, padding=0)
                    self.plot_widget.setYRange(y_min, y_max, padding=0)
                    
                    # Only set X range if not updating from auto-load (to preserve user's pan position)
                    if not self.updating_view:
                        # Set X range to show the requested view range (in seconds)
                        x_padding = max((all_x_max - all_x_min) * 0.1, 0.1)  # At least 0.1 seconds
                        x_min = all_x_min - x_padding
                        x_max = all_x_max + x_padding
                        vb.setXRange(x_min, x_max, padding=0)
                        self.plot_widget.setXRange(x_min, x_max, padding=0)
                    
                    print(f"Set plot range: y=[{y_min:.1f}, {y_max:.1f}], "
                          f"x-range preserved from user view" if self.updating_view else f"x=[{all_x_min-x_padding:.1f}, {all_x_max+x_padding:.1f}]")
                    print(f"  Auto-range disabled, ViewBox auto-visible disabled")
                else:
                    # Fallback to autoRange if no valid data
                    self.plot_widget.enableAutoRange()
                    self.plot_widget.autoRange()
                    print("Using autoRange (no valid data found)")
            else:
                self.plot_widget.enableAutoRange()
                self.plot_widget.autoRange()
            
            # Force GUI update
            QApplication.processEvents()
            
            # Try to trigger a repaint
            self.plot_widget.repaint()
            self.plot_widget.update()
            
            print(f"Plot widget updated. Number of items: {len(self.plot_widget.listDataItems())}")
        
        # Force update again
        QApplication.processEvents()
    
    def update_spectrogram_view(self):
        """Update the spectrogram view."""
        if self.current_data is None or not HAS_SCIPY or len(self.selected_channels) == 0:
            return
        
        try:
            # Compute spectrogram for first selected channel
            channel_data = self.current_data[:, 0]
            
            # Ensure we have enough data for spectrogram
            if len(channel_data) < 128:
                return  # Not enough data for spectrogram
            
            nperseg = min(2048, len(channel_data) // 4)
            nperseg = max(64, nperseg)  # Minimum segment size
            noverlap = nperseg // 2
            
            f, t, Sxx = spectrogram(
                channel_data, fs=self.sampling_rate,
                nperseg=nperseg, noverlap=noverlap,
                window='hann'
            )
            
            # Convert to dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # Update ImageView - simpler approach
            self.spectrogram_viewer.setImage(Sxx_db, autoRange=True)
            # Labels are set automatically by ImageView, skip manual label setting
            
        except Exception as e:
            print(f"Error updating spectrogram: {e}")
    
    def update_psd_view(self):
        """Update the power spectral density plot."""
        if self.current_data is None or not HAS_SCIPY:
            return
        
        self.psd_widget.clear()
        
        try:
            # Generate colors - use color names for maximum visibility
            color_palette = ['w', 'r', 'g', 'b', 'c', 'm', 'y', '#FFA500', '#FF00FF', '#00FFFF']  # white, red, green, blue, cyan, magenta, yellow, orange, fuchsia, aqua
            
            for i, ch_idx in enumerate(self.selected_channels):
                channel_data = self.current_data[:, i]
                
                # Compute PSD using Welch's method
                if len(channel_data) < 128:
                    continue  # Not enough data for PSD
                
                nperseg = min(2048, len(channel_data) // 4)
                nperseg = max(64, nperseg)  # Minimum segment size
                f, Pxx = welch(
                    channel_data, fs=self.sampling_rate,
                    nperseg=nperseg, window='hann'
                )
                
                # Convert to dB
                Pxx_db = 10 * np.log10(Pxx + 1e-10)
                
                # Cycle through color palette - use bright colors
                color = color_palette[i % len(color_palette)]
                pen = pg.mkPen(color=color, width=2)
                self.psd_widget.plot(
                    f, Pxx_db,
                    pen=pen,
                    name=f'Ch {ch_idx}'
                )
            
            self.psd_widget.setLabel('bottom', 'Frequency (Hz)')
            self.psd_widget.autoRange()
            
        except Exception as e:
            print(f"Error updating PSD: {e}")
    
    def apply_filter(self):
        """Apply filter to current data."""
        if self.current_data is None or not HAS_SCIPY:
            QMessageBox.warning(self, "Warning", "No data loaded or scipy not available")
            return
        
        filter_type = self.filter_type_combo.currentText()
        
        # Get filter parameters
        params = {
            'order': self.filter_order_spin.value()
        }
        
        if filter_type == 'bandpass':
            params['low'] = self.filter_low_spin.value()
            params['high'] = self.filter_high_spin.value()
        elif filter_type == 'highpass':
            params['cutoff'] = self.filter_low_spin.value()
        elif filter_type == 'lowpass':
            params['cutoff'] = self.filter_high_spin.value()
        elif filter_type == 'notch':
            params['freq'] = self.filter_notch_spin.value()
            params['quality'] = 30.0
        
        # Add to filter chain
        filter_entry = {
            'type': filter_type,
            'params': params.copy()
        }
        self.filter_chain.append(filter_entry)
        self.update_filter_chain_label()
        
        # Apply filter
        self.statusBar().showMessage("Applying filter...")
        QApplication.processEvents()
        
        try:
            worker = FilterWorker(self.current_data, filter_type, params, self.sampling_rate)
            worker.finished.connect(self.on_filter_finished)
            worker.error.connect(self.on_filter_error)
            worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filter: {e}")
    
    def on_filter_finished(self, filtered_data: np.ndarray):
        """Handle filter completion."""
        self.current_data = filtered_data
        self.update_time_series_view()
        if HAS_SCIPY:
            self.update_spectrogram_view()
            self.update_psd_view()
        self.statusBar().showMessage("Filter applied successfully")
    
    def on_filter_error(self, error_msg: str):
        """Handle filter error."""
        QMessageBox.critical(self, "Filter Error", error_msg)
        self.statusBar().showMessage("Filter error occurred")
    
    def clear_filter(self):
        """Clear all filters and reload raw data."""
        self.filter_chain.clear()
        self.update_filter_chain_label()
        
        if self.raw_data is not None:
            self.current_data = self.raw_data.copy()
            self.apply_processing(internal=True)
            self.update_time_series_view()
            if HAS_SCIPY:
                self.update_spectrogram_view()
                self.update_psd_view()
    
    def update_filter_chain_label(self):
        """Update the filter chain label."""
        if not self.filter_chain:
            self.filter_chain_label.setText("Filters: None")
        else:
            chain_text = "Filters: " + ", ".join([
                f"{f['type']}({', '.join(f'{k}={v}' for k, v in f['params'].items())})"
                for f in self.filter_chain
            ])
            self.filter_chain_label.setText(chain_text)
    
    def on_processing_changed(self):
        """Handle processing option changes."""
        pass  # Will be applied when loading/updating
    
    def apply_processing(self, internal: bool = False):
        """Apply signal processing options."""
        if self.current_data is None:
            return
        
        processed = self.current_data.copy()
        
        # DC removal
        if self.dc_remove_check.isChecked():
            processed = processed - processed.mean(axis=0, keepdims=True)
        
        # Normalization
        if self.normalize_check.isChecked():
            std = processed.std(axis=0, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            processed = processed / std
        
        # Rereferencing (apply AFTER normalization if needed, to preserve signal)
        reref_method = self.reref_combo.currentText()
        if reref_method == 'Average':
            # Reference to average of all selected channels
            # This makes all channels sum to zero, so they'll be very small
            avg_ref = processed.mean(axis=1, keepdims=True)
            processed = processed - avg_ref
        elif reref_method == 'Common Average':
            # Common average reference (CAR) - same as Average for now
            if self.loader and len(self.selected_channels) > 1:
                avg_ref = processed.mean(axis=1, keepdims=True)
                processed = processed - avg_ref
        
        self.current_data = processed
        
        if not internal:
            self.update_time_series_view()
            if HAS_SCIPY:
                self.update_spectrogram_view()
                self.update_psd_view()
    
    def zoom_to_full(self):
        """Zoom to show full data range."""
        if self.loader is None:
            return
        
        self.start_sample = 0
        self.end_sample = self.loader.num_samples_per_channel
        self.start_sample_spin.setValue(self.start_sample)
        self.end_sample_spin.setValue(self.end_sample)
        self.load_and_update_view()
    
    def reset_view(self):
        """Reset the plot view."""
        self.plot_widget.autoRange()
    
    def export_data(self):
        """Export processed data to file."""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data to export")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Processed Data", "", "NPY files (*.npy);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not filepath:
            return
        
        try:
            filepath = Path(filepath)
            
            if filepath.suffix == '.npy':
                np.save(filepath, self.current_data)
            elif filepath.suffix == '.csv':
                np.savetxt(filepath, self.current_data, delimiter=',', fmt='%.6f')
            else:
                # Default to npy
                filepath = filepath.with_suffix('.npy')
                np.save(filepath, self.current_data)
            
            QMessageBox.information(self, "Success", f"Data exported to:\n{filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {e}")
    
    def export_metadata(self):
        """Export metadata about the current session."""
        if self.loader is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Metadata", "", "JSON files (*.json);;All files (*.*)"
        )
        
        if not filepath:
            return
        
        try:
            metadata = {
                'channels': self.selected_channels,
                'num_channels': len(self.selected_channels),
                'start_sample': self.start_sample,
                'end_sample': self.end_sample,
                'num_samples': self.end_sample - self.start_sample,
                'sampling_rate': self.sampling_rate,
                'duration_seconds': (self.end_sample - self.start_sample) / self.sampling_rate,
                'filter_chain': self.filter_chain,
                'processing': {
                    'dc_removal': self.dc_remove_check.isChecked(),
                    'normalization': self.normalize_check.isChecked(),
                    'rereferencing': self.reref_combo.currentText()
                },
                'export_timestamp': datetime.now().isoformat(),
                'source_file': str(self.loader.filepath)
            }
            
            filepath = Path(filepath)
            if filepath.suffix != '.json':
                filepath = filepath.with_suffix('.json')
            
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            QMessageBox.information(self, "Success", f"Metadata exported to:\n{filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export metadata: {e}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = AdvancedEEGGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

