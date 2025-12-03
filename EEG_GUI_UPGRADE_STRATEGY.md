# EEG Analysis GUI - Comprehensive Upgrade Strategy

## 🎯 Project Goal
Transform current EEG viewer into a production-quality, modular EEG analysis platform inspired by Open Ephys GUI, optimized for seizure detection and multi-channel EEG visualization.

---

## 📊 Current State Analysis

### ✅ What Works Well
1. **PyQtGraph-based visualization** - Good foundation for performance
2. **Auto-loading with buffer** - Smart data management
3. **Sample/second coordinate system** - Correct approach
4. **Overview/minimap widget** - Good UX feature
5. **Multiple views** (time series, spectrogram, PSD) - Good feature set
6. **Filter chain tracking** - Good for reproducibility

### ⚠️ Issues Identified

#### **1. Channel Plotting Performance**
- **Current**: Plots all channels sequentially with offsets
- **Problem**: 
  - Becomes slow with many channels (>20)
  - Excessive replotting on every update
  - Not using PyQtGraph's optimizations effectively
- **Impact**: Sluggish when scrolling/zooming with full 72-channel data

#### **2. Downsampling Logic**
```python
# Current approach (lines 1080-1130)
if visible_range_seconds < 2.0:
    # Full resolution - GOOD
    plot_data = extract visible range
else:
    # Adaptive downsampling - NEEDS VERIFICATION
    downsample_factor = max(1, plot_data.shape[0] // max_points)
    plot_data = plot_data[::downsample_factor]  # Simple decimation
```
- **Problem**: Simple decimation can cause aliasing
- **Better approach**: Use proper anti-aliasing downsampling

#### **3. No Annotation System**
- Missing trigger detection
- No seizure marking capabilities
- No refractory period handling
- No event visualization

#### **4. No Bad Channel Marking**
- Can't mark channels as bad interactively
- No visual indication of bad channels
- No automatic bad channel detection

#### **5. Channel Selection UX**
- Basic list widget
- No grouping or searching
- No quick select patterns (e.g., "all frontal")

#### **6. Scale Controls**
- X-scale changes entire window (not intuitive for users)
- Y-scale is global only (no per-channel adjustment)

---

## 🏗️ Architecture Strategy: Modular Design (Open Ephys-inspired)

### Module Categories

#### **1. SOURCES**
```
┌─────────────────────┐
│  File Reader        │  - Load continuous.dat, EDF, etc.
│  Live Stream        │  - Future: real-time acquisition
│  Network Stream     │  - Future: LSL/OSC input
└─────────────────────┘
```

#### **2. PROCESSORS (Filter Chain)**
```
┌─────────────────────┐
│  Bandpass Filter    │
│  Notch Filter       │
│  Rereferencing      │
│  Artifact Rejection │
│  Downsampling       │
└─────────────────────┘
```

#### **3. DETECTORS**
```
┌─────────────────────┐
│  Trigger Detector   │  - For seizure markers (with refractory period)
│  Spike Detector     │  - For spike-wave detection
│  Artifact Detector  │  - Automatic bad channel/segment detection
└─────────────────────┘
```

#### **4. VISUALIZERS**
```
┌─────────────────────┐
│  Raw Traces         │  - Main multi-channel view
│  Event Raster       │  - Seizure/event timeline
│  Spectrogram        │  - Time-frequency analysis
│  Topography         │  - Spatial maps
└─────────────────────┘
```

#### **5. UTILITIES**
```
┌─────────────────────┐
│  Annotation Editor  │
│  Channel Manager    │
│  Export Manager     │
└─────────────────────┘
```

---

## 🎨 GUI Layout Design

```
┌─────────────────────────────────────────────────────────────────┐
│  MenuBar  │  [▶] [⏸] [⏹]  │  Time: 00:04:32 │  Latency: 23ms  │
├─────────┬─────────────────────────────────────────────────────┤
│         │                                                       │
│  MODULE │              MAIN VISUALIZATION AREA                 │
│  PANEL  │         (Multi-channel EEG Traces)                   │
│         │                                                       │
│ Sources │  ┌───────────────────────────────────────────────┐  │
│ Filters │  │ Ch1  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      │  │
│ Detectors│  │ Ch2  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      │  │
│ Display │  │ Ch3  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      │  │
│         │  │ ...                                            │  │
│         │  └───────────────────────────────────────────────┘  │
│         │         (Overview/Minimap)                           │
│         │  ┌───────────────────────────────────────────────┐  │
│         │  │ ▼                                              │  │
│         │  └───────────────────────────────────────────────┘  │
│         │                                                       │
│         │              EVENT RASTER VIEW                       │
│         │  ┌───────────────────────────────────────────────┐  │
│         │  │ Seizures: ■     ■           ■                 │  │
│         │  │ Triggers: ||| | ||  | ||||                    │  │
│         │  └───────────────────────────────────────────────┘  │
├─────────┴─────────────────────────────────────────────────────┤
│              EXPANDABLE MODULE CONFIGURATION PANEL            │
│  [Trigger Detector] [Bad Ch Manager] [Filter Settings]        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementation Priority List

### **Phase 1: Core Performance & Plotting (CRITICAL)**
**Goal**: Make visualization buttery smooth with 72 channels

#### 1.1 Optimize Channel Plotting ⭐⭐⭐
**Current Problem**: Sequential plotting is slow
**Solution**: 
```python
# Use PyQtGraph's MultiPlotWidget or individual PlotItems
# Pre-create all PlotItems once, then update data
class OptimizedChannelPlot:
    def __init__(self):
        self.plot_items = []  # Pre-create
        self.curves = []      # Store curve references
        
    def create_channels(self, n_channels):
        """Create all plot items once"""
        for i in range(n_channels):
            plot_item = self.plot_widget.addPlot(row=i, col=0)
            curve = plot_item.plot(pen=color)
            self.plot_items.append(plot_item)
            self.curves.append(curve)
    
    def update_data(self, data):
        """Just update curve data, don't recreate"""
        for i, curve in enumerate(self.curves):
            curve.setData(x=time_axis, y=data[:, i])
```

**References**: 
- MNE-Qt-Browser: https://github.com/mne-tools/mne-qt-browser
- PyQtGraph performance guide

#### 1.2 Fix Downsampling with Anti-Aliasing ⭐⭐⭐
**Current Problem**: Simple decimation causes aliasing
**Solution**:
```python
def downsample_with_antialiasing(data, downsample_factor, axis=0):
    """Proper downsampling with anti-aliasing using scipy.signal.decimate"""
    if downsample_factor <= 1:
        return data
    
    # Use scipy.signal.decimate which includes anti-aliasing filter
    from scipy.signal import decimate
    
    # Decimate along time axis (axis 0)
    downsampled = decimate(data, downsample_factor, axis=axis, 
                           zero_phase=True, ftype='fir')
    return downsampled
```

**Test**: Compare with MNE's `raw.resample()` method

#### 1.3 Implement Proper Data Caching ⭐⭐
**Goal**: Cache multiple zoom levels like Google Maps
```python
class MultiResolutionCache:
    """Store data at multiple resolutions"""
    def __init__(self):
        self.cache = {
            'full': None,      # Full resolution
            '10x': None,       # 10x downsampled
            '100x': None,      # 100x downsampled
            '1000x': None,     # For overview
        }
    
    def get_data_for_zoom(self, zoom_level):
        """Return appropriate resolution for current zoom"""
        pass
```

### **Phase 2: Annotation System (HIGH PRIORITY)** ⭐⭐⭐
**Goal**: Detect and annotate seizure events

#### 2.1 Trigger Detection with Refractory Period
```python
class TriggerDetector:
    """Detect trigger pulses with refractory period"""
    
    def __init__(self, threshold=0.5, refractory_period=21.0, 
                 sampling_rate=200.0):
        self.threshold = threshold
        self.refractory_samples = int(refractory_period * sampling_rate)
        self.last_trigger = -np.inf
    
    def detect_triggers(self, trigger_channel):
        """
        Detect rising edge crossings of threshold with refractory period
        
        Args:
            trigger_channel: 1D array of trigger data
            
        Returns:
            trigger_times: Array of sample indices where triggers occurred
        """
        # Detect threshold crossings
        above_threshold = trigger_channel > self.threshold
        rising_edges = np.diff(above_threshold.astype(int)) > 0
        
        trigger_samples = np.where(rising_edges)[0]
        
        # Apply refractory period
        valid_triggers = []
        last_trigger = -self.refractory_samples
        
        for trigger in trigger_samples:
            if trigger - last_trigger >= self.refractory_samples:
                valid_triggers.append(trigger)
                last_trigger = trigger
        
        return np.array(valid_triggers)
```

#### 2.2 Annotation Data Structure
```python
@dataclass
class Annotation:
    """Represent a single annotation"""
    start_sample: int
    end_sample: int
    label: str
    description: str = ""
    color: str = "red"
    confidence: float = 1.0

class AnnotationManager:
    """Manage all annotations"""
    def __init__(self):
        self.annotations = []
    
    def add_annotation(self, start, end, label):
        """Add new annotation"""
        pass
    
    def get_annotations_in_range(self, start, end):
        """Get annotations overlapping time range"""
        pass
    
    def save_annotations(self, filepath):
        """Save to JSON/CSV"""
        pass
```

#### 2.3 Visual Annotation Overlay
- Draw colored regions on time series plot
- Add annotation track below main view
- Allow drag-to-create annotations
- Allow drag-to-edit annotation boundaries

### **Phase 3: Bad Channel Management** ⭐⭐

#### 3.1 Interactive Bad Channel Marking
```python
class ChannelManager:
    """Manage channel states and selection"""
    
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.bad_channels = set()
        self.channel_groups = {}  # e.g., {'frontal': [0,1,2,3], ...}
    
    def mark_bad(self, channel_idx):
        """Mark channel as bad"""
        self.bad_channels.add(channel_idx)
    
    def is_bad(self, channel_idx):
        """Check if channel is bad"""
        return channel_idx in self.bad_channels
```

**UI Features**:
- Right-click channel label → "Mark as bad"
- Bad channels shown in red/gray
- Automatically exclude from processing
- Save bad channels list with data

#### 3.2 Automatic Bad Channel Detection
```python
def detect_bad_channels(data, sampling_rate):
    """
    Detect bad channels based on statistical properties
    
    Criteria:
    - Flat line (std < threshold)
    - Too much variance (std > threshold)
    - Correlation with other channels too low
    - Excessive high-frequency noise
    """
    bad_channels = []
    
    for ch in range(data.shape[1]):
        ch_data = data[:, ch]
        
        # Check for flat line
        if np.std(ch_data) < 0.1:
            bad_channels.append(ch)
            continue
        
        # Check for excessive variance
        if np.std(ch_data) > 1000:
            bad_channels.append(ch)
            continue
        
        # More sophisticated checks...
    
    return bad_channels
```

### **Phase 4: Enhanced Channel Selection** ⭐⭐

#### 4.1 Improved Channel List Widget
```python
class AdvancedChannelSelector(QWidget):
    """Enhanced channel selection with grouping and search"""
    
    def __init__(self):
        super().__init__()
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search channels...")
        
        # Tree widget for hierarchical display
        self.tree = QTreeWidget()
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        
        # Groups
        self.groups = {
            'Frontal': [0, 1, 2, 3],
            'Temporal': [4, 5, 6, 7],
            'Parietal': [8, 9, 10, 11],
            # ...
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create hierarchical channel list"""
        for group_name, channels in self.groups.items():
            group_item = QTreeWidgetItem([group_name])
            for ch in channels:
                ch_item = QTreeWidgetItem([f"Channel {ch}"])
                group_item.addChild(ch_item)
            self.tree.addTopLevelItem(group_item)
```

**Features**:
- Search/filter channels
- Group by region (frontal, temporal, etc.)
- Quick select: "All frontal", "All even", etc.
- Show channel impedance/quality indicator

### **Phase 5: Better Scale Controls** ⭐

#### 5.1 Intuitive Scale Behavior
**Current**: X-scale changes window width (confusing)
**Better**: 
- X-scale: Stretch/compress visible traces horizontally
- Y-scale: Adjust amplitude scaling
- Keep window width constant

```python
def apply_y_scale(self, scale_factor):
    """Scale amplitude without changing offset"""
    for i, curve in enumerate(self.curves):
        y_data = self.original_data[:, i] * scale_factor
        y_data = y_data + (i * self.channel_spacing)
        curve.setData(y=y_data)
```

#### 5.2 Per-Channel Scale
- Allow individual channel scaling
- Store scale factors per channel
- Reset to default scale

---

## 📐 Data Flow Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                         │
│  [File Reader] [Live Stream] [Network]                     │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                     RAW DATA BUFFER                         │
│  - Stores original data                                     │
│  - Memory-mapped for large files                            │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                   PROCESSOR CHAIN                           │
│  [Filter] → [Rereference] → [Detect Bad Ch] → ...         │
│  - Each processor is modular                                │
│  - Can be enabled/disabled                                  │
│  - Settings stored in processor                             │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────┐
│                  PROCESSED DATA CACHE                       │
│  - Multi-resolution storage                                 │
│  - LRU cache for memory management                          │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ├──────────┬──────────┬──────────┐
                 ▼          ▼          ▼          ▼
         ┌──────────┐ ┌─────────┐ ┌──────┐ ┌────────┐
         │Raw Traces│ │ Raster  │ │ Spec │ │  Topo  │
         └──────────┘ └─────────┘ └──────┘ └────────┘
```

---

## 🚀 Best Practices from Leading Tools

### From MNE-Qt-Browser
1. **Separate plot items per channel** - Much faster than single plot with offsets
2. **Update data, not recreate plots** - Use `curve.setData()` not `plot()`
3. **Viewport-based rendering** - Only render what's visible
4. **Annotation objects** - Separate overlay layer for annotations

### From Visbrain Sleep
1. **GPU acceleration via VisPy** - Consider for future if PyQtGraph isn't fast enough
2. **Keyboard shortcuts** - Essential for power users
3. **Reversible operations** - Never modify raw data
4. **Settings persistence** - Save/load user preferences

### From Open Ephys
1. **Modular architecture** - Each processor is independent
2. **Visual signal chain** - User can see data flow
3. **Real-time indicators** - CPU usage, latency, etc.
4. **Expandable panels** - Configuration panels that expand/collapse

---

## 🧪 Testing Strategy

### Performance Benchmarks
```python
def benchmark_plotting():
    """Measure plotting performance"""
    n_channels = 72
    n_samples = 10000
    
    # Test 1: Plot creation time
    start = time.time()
    create_plots(n_channels)
    print(f"Plot creation: {time.time() - start:.3f}s")
    
    # Test 2: Data update time
    start = time.time()
    update_data(data)
    print(f"Data update: {time.time() - start:.3f}s")
    
    # Test 3: Scroll lag
    # Should be < 16ms for 60 FPS
```

### Validation Tests
- Verify downsampling doesn't lose important features
- Compare trigger detection with manual annotations
- Test bad channel detection on known datasets
- Verify annotation save/load

---

## 📋 Implementation Checklist

### Week 1: Core Performance
- [ ] Refactor channel plotting to use persistent plot items
- [ ] Implement proper anti-aliasing downsampling
- [ ] Add multi-resolution data caching
- [ ] Benchmark and optimize to < 100ms update time for 72 channels

### Week 2: Annotation System
- [ ] Implement trigger detection with refractory period
- [ ] Create annotation data structure
- [ ] Add visual annotation overlay
- [ ] Implement annotation editor UI
- [ ] Add annotation save/load (JSON/CSV)

### Week 3: Bad Channel Management
- [ ] Add right-click menu for channel marking
- [ ] Implement bad channel visual indicators
- [ ] Add automatic bad channel detection
- [ ] Integrate bad channels with processing pipeline

### Week 4: Enhanced UI
- [ ] Create advanced channel selector with search
- [ ] Add keyboard shortcuts
- [ ] Implement better scale controls
- [ ] Add event raster view
- [ ] Create expandable configuration panels

### Week 5: Polish & Testing
- [ ] Add settings persistence
- [ ] Performance optimization
- [ ] User documentation
- [ ] Testing with real seizure data

---

## 🎯 Success Metrics

1. **Performance**: 
   - < 100ms update latency for 72 channels
   - Smooth scrolling at 60 FPS
   - Can handle hours of data without lag

2. **Usability**:
   - Can annotate 100 seizures in < 10 minutes
   - Bad channel marking takes < 5 seconds
   - New users productive within 15 minutes

3. **Reliability**:
   - No crashes during 8-hour sessions
   - Annotations never lost
   - Reproducible analysis results

---

## 📚 Key References

1. **MNE-Qt-Browser**: https://github.com/mne-tools/mne-qt-browser
   - Study: `_mpl_figure.py` for plotting optimization
   - Study: `_pg_figure.py` for PyQtGraph implementation

2. **Visbrain Sleep**: https://github.com/EtienneCmb/visbrain
   - Study: `visbrain/gui/sleep/interface/ui_elements.py`
   - Study: GPU acceleration techniques

3. **PyQtGraph Examples**: 
   - `MultiPlotWidget` example
   - Infinite scrolling example
   - Real-time plotting

4. **Open Ephys GUI** (C++, but good design reference):
   - Modular architecture pattern
   - Visual processor chain

---

## 🔄 Migration Path

### Phase 1: Keep current code working
- Create new modules alongside existing code
- Add feature flags to switch between old/new

### Phase 2: Gradual replacement
- Replace plotting engine first (most critical)
- Add annotation system as separate module
- Integrate bad channel management

### Phase 3: Full transition
- Remove old plotting code
- Unified architecture
- Full documentation

---

## 💡 Quick Wins (Do First!)

1. **Use `curve.setData()` instead of `plot()` repeatedly** 
   - 10x speedup immediately
   - 1 hour to implement

2. **Add keyboard shortcuts for navigation**
   - Arrow keys: pan
   - +/- : zoom
   - Space: mark event
   - 2 hours to implement

3. **Right-click context menu on channels**
   - Mark as bad
   - Hide channel
   - Change color
   - 1 hour to implement

4. **Status bar with useful info**
   - Current time
   - Selected range
   - Number of annotations
   - 30 minutes to implement

---

## Next Steps

1. **Review this strategy** - Discuss priorities
2. **Set up Claude Code environment** - Ensure all dependencies available
3. **Start with Phase 1.1** - Optimize channel plotting first
4. **Test with your real data** - Validate improvements
5. **Iterate** - Adjust strategy based on results

Ready to proceed? Let's start with the plotting optimization!
