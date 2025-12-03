# EEG GUI - Quick Start Implementation Guide

## 🎯 Immediate Priority: Fix Core Performance Issues

### Problem Summary
Your current script has 3 critical bottlenecks:

1. **Plotting is slow** - Recreates all plots on every update
2. **Downsampling might alias** - Simple decimation without anti-aliasing
3. **No annotations** - Critical for your seizure work

---

## 🚀 Quick Win #1: Fix Plotting Performance (1-2 hours)

### Current Problem
```python
# In update_time_series_view() - lines ~1200-1300
self.plot_widget.clear()  # ⚠️ CLEARS EVERYTHING
for i, ch_idx in enumerate(self.selected_channels):
    # Creates new plot every time ⚠️
    plot_item = pg.PlotDataItem(time_array, y_array, pen=pen)
    self.plot_widget.addItem(plot_item)  # ⚠️ SLOW
```

**Impact**: Every scroll/zoom recreates 72 plots = SLOW

### Solution: Persistent Curves
```python
class OptimizedEEGPlot:
    """Fast EEG plotting using persistent curves"""
    
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self.curves = {}  # Store curve objects by channel
        self.colors = self._generate_colors()
    
    def initialize_channels(self, channel_indices):
        """Create curves once - call this when channels change"""
        self.plot_widget.clear()
        self.curves.clear()
        
        for i, ch_idx in enumerate(channel_indices):
            # Create curve ONCE
            curve = pg.PlotDataItem(
                pen=pg.mkPen(self.colors[i % len(self.colors)], width=2),
                name=f'Ch {ch_idx}'
            )
            self.plot_widget.addItem(curve)
            self.curves[ch_idx] = curve
    
    def update_data(self, time_array, data_dict, channel_spacing):
        """Update existing curves - FAST!"""
        for i, (ch_idx, curve) in enumerate(self.curves.items()):
            if ch_idx in data_dict:
                y_data = data_dict[ch_idx] + (i * channel_spacing)
                # Just update data - NO recreation
                curve.setData(time_array, y_data)
    
    def _generate_colors(self):
        """Nice color palette"""
        return ['w', 'r', 'g', 'b', 'c', 'm', 'y', 
                '#FFA500', '#FF00FF', '#00FFFF']
```

**Expected Result**: 10-50x faster updates!

---

## 🚀 Quick Win #2: Add Keyboard Shortcuts (30 min)

```python
def keyPressEvent(self, event):
    """Handle keyboard shortcuts"""
    key = event.key()
    modifiers = event.modifiers()
    
    # Navigation
    if key == Qt.Key.Key_Left:
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.navigate_time(-10000)  # Shift+Left: -10k samples
        else:
            self.navigate_time(-1000)   # Left: -1k samples
    
    elif key == Qt.Key.Key_Right:
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            self.navigate_time(10000)
        else:
            self.navigate_time(1000)
    
    # Zoom
    elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
        self.increase_x_scale()
    elif key == Qt.Key.Key_Minus:
        self.decrease_x_scale()
    
    # Amplitude
    elif key == Qt.Key.Key_Up:
        self.increase_y_scale()
    elif key == Qt.Key.Key_Down:
        self.decrease_y_scale()
    
    # Mark annotation (Space bar)
    elif key == Qt.Key.Key_Space:
        self.quick_mark_annotation()
    
    # Toggle bad channel (B key when channel selected)
    elif key == Qt.Key.Key_B:
        self.toggle_bad_channel_at_cursor()
    
    # Home/End
    elif key == Qt.Key.Key_Home:
        self.go_to_start()
    elif key == Qt.Key.Key_End:
        self.go_to_end()
    
    else:
        super().keyPressEvent(event)
```

---

## 🚀 Quick Win #3: Trigger Detection (1-2 hours)

### Implementation
```python
class TriggerDetector:
    """Detect trigger pulses for seizure markers"""
    
    def __init__(self, threshold=0.5, refractory_seconds=21.0, 
                 sampling_rate=200.0):
        self.threshold = threshold
        self.refractory_samples = int(refractory_seconds * sampling_rate)
        self.sampling_rate = sampling_rate
        self.detected_triggers = []
    
    def detect(self, trigger_channel_data, start_sample=0):
        """
        Detect triggers with refractory period
        
        Args:
            trigger_channel_data: 1D array of trigger channel
            start_sample: Offset for sample indices
            
        Returns:
            List of trigger sample indices
        """
        # Threshold crossing detection
        above = trigger_channel_data > self.threshold
        
        # Find rising edges
        rising = np.diff(above.astype(int)) > 0
        trigger_samples = np.where(rising)[0] + start_sample
        
        # Apply refractory period
        if len(trigger_samples) == 0:
            return []
        
        filtered_triggers = [trigger_samples[0]]
        
        for trigger in trigger_samples[1:]:
            if trigger - filtered_triggers[-1] >= self.refractory_samples:
                filtered_triggers.append(trigger)
        
        return filtered_triggers
    
    def plot_triggers(self, plot_widget, trigger_samples, color='red'):
        """Draw vertical lines at trigger times"""
        for sample in trigger_samples:
            time = sample / self.sampling_rate
            line = pg.InfiniteLine(
                pos=time, 
                angle=90, 
                pen=pg.mkPen(color, width=2, style=Qt.PenStyle.DashLine),
                label='Trigger'
            )
            plot_widget.addItem(line)

# Usage in your main class:
def detect_and_plot_triggers(self):
    """Detect triggers and show them on plot"""
    if self.loader is None or not hasattr(self, 'trigger_channel_idx'):
        return
    
    # Load trigger channel
    trigger_data = self.loader.load_channels(
        [self.trigger_channel_idx],
        start_sample=self.start_sample,
        end_sample=self.end_sample
    )[:, 0]
    
    # Detect triggers
    detector = TriggerDetector(
        threshold=self.trigger_threshold,  # Make this adjustable
        refractory_seconds=21.0,
        sampling_rate=self.sampling_rate
    )
    
    triggers = detector.detect(trigger_data, self.start_sample)
    
    # Plot them
    detector.plot_triggers(self.plot_widget, triggers)
    
    # Store for annotation
    self.current_triggers = triggers
```

### Add UI Controls
```python
def create_trigger_detection_group(self):
    """Create trigger detection controls"""
    group = QGroupBox("Trigger Detection")
    layout = QVBoxLayout()
    
    # Trigger channel selection
    ch_layout = QHBoxLayout()
    ch_layout.addWidget(QLabel("Trigger Channel:"))
    self.trigger_channel_spin = QSpinBox()
    self.trigger_channel_spin.setMaximum(self.loader.num_channels if self.loader else 72)
    ch_layout.addWidget(self.trigger_channel_spin)
    layout.addLayout(ch_layout)
    
    # Threshold
    thresh_layout = QHBoxLayout()
    thresh_layout.addWidget(QLabel("Threshold:"))
    self.trigger_threshold_spin = QDoubleSpinBox()
    self.trigger_threshold_spin.setRange(0.0, 10.0)
    self.trigger_threshold_spin.setValue(0.5)
    self.trigger_threshold_spin.setSingleStep(0.1)
    thresh_layout.addWidget(self.trigger_threshold_spin)
    layout.addLayout(thresh_layout)
    
    # Refractory period
    refrac_layout = QHBoxLayout()
    refrac_layout.addWidget(QLabel("Refractory (s):"))
    self.refractory_spin = QDoubleSpinBox()
    self.refractory_spin.setRange(0.0, 60.0)
    self.refractory_spin.setValue(21.0)
    refrac_layout.addWidget(self.refractory_spin)
    layout.addLayout(refrac_layout)
    
    # Detect button
    detect_btn = QPushButton("Detect Triggers")
    detect_btn.clicked.connect(self.detect_and_plot_triggers)
    layout.addWidget(detect_btn)
    
    # Results label
    self.trigger_results_label = QLabel("No triggers detected")
    layout.addWidget(self.trigger_results_label)
    
    group.setLayout(layout)
    return group
```

---

## 🚀 Quick Win #4: Bad Channel Marking (1 hour)

### Add Context Menu
```python
def setup_channel_list_context_menu(self):
    """Add right-click menu to channel list"""
    self.channel_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.channel_list.customContextMenuRequested.connect(self.show_channel_context_menu)

def show_channel_context_menu(self, position):
    """Show context menu for channel"""
    item = self.channel_list.itemAt(position)
    if item is None:
        return
    
    channel_idx = self.channel_list.row(item)
    
    menu = QMenu()
    
    # Mark as bad action
    if channel_idx in self.bad_channels:
        mark_action = menu.addAction("Mark as Good")
    else:
        mark_action = menu.addAction("Mark as Bad")
    
    # Hide channel action
    hide_action = menu.addAction("Hide Channel")
    
    # Change color action
    color_action = menu.addAction("Change Color...")
    
    # Execute menu
    action = menu.exec(self.channel_list.mapToGlobal(position))
    
    if action == mark_action:
        self.toggle_bad_channel(channel_idx)
    elif action == hide_action:
        self.hide_channel(channel_idx)
    elif action == color_action:
        self.change_channel_color(channel_idx)

def toggle_bad_channel(self, channel_idx):
    """Toggle bad channel status"""
    if not hasattr(self, 'bad_channels'):
        self.bad_channels = set()
    
    if channel_idx in self.bad_channels:
        self.bad_channels.remove(channel_idx)
        print(f"Channel {channel_idx} marked as GOOD")
    else:
        self.bad_channels.add(channel_idx)
        print(f"Channel {channel_idx} marked as BAD")
    
    # Update visual indication
    item = self.channel_list.item(channel_idx)
    if channel_idx in self.bad_channels:
        item.setForeground(Qt.GlobalColor.red)
        item.setText(f"Channel {channel_idx:3d} [BAD]")
    else:
        item.setForeground(Qt.GlobalColor.white)
        item.setText(f"Channel {channel_idx:3d}")
    
    # Redraw if channel is currently displayed
    if channel_idx in self.selected_channels:
        self.update_time_series_view()
```

---

## 🧪 Testing Your Improvements

### Performance Test
```python
import time

def benchmark_plotting_performance():
    """Measure plotting speed"""
    n_updates = 100
    
    # Old method
    start = time.time()
    for _ in range(n_updates):
        plot_widget.clear()
        for ch in channels:
            plot_widget.plot(...)  # Old way
    old_time = time.time() - start
    
    # New method
    start = time.time()
    for _ in range(n_updates):
        for curve in curves:
            curve.setData(...)  # New way
    new_time = time.time() - start
    
    print(f"Old: {old_time:.2f}s ({old_time/n_updates*1000:.1f}ms per update)")
    print(f"New: {new_time:.2f}s ({new_time/n_updates*1000:.1f}ms per update)")
    print(f"Speedup: {old_time/new_time:.1f}x")
```

---

## 📋 Implementation Order

### Day 1: Performance
1. ✅ Refactor plotting to use persistent curves
2. ✅ Add keyboard shortcuts
3. ✅ Test with 72 channels

### Day 2: Trigger Detection  
1. ✅ Implement TriggerDetector class
2. ✅ Add UI controls
3. ✅ Test with real data

### Day 3: Bad Channels
1. ✅ Add context menu
2. ✅ Visual indicators
3. ✅ Save/load bad channels

### Day 4: Polish
1. ✅ Add status bar info
2. ✅ Improve color scheme
3. ✅ User testing

---

## 🎯 Success Criteria

After these quick wins, you should have:

✅ **Smooth scrolling** - No lag with 72 channels
✅ **Keyboard navigation** - Arrow keys, +/-, space
✅ **Trigger detection** - Finds seizure events with refractory period
✅ **Bad channel marking** - Right-click to mark/unmark
✅ **Better UX** - Faster, more intuitive

**Total time**: ~6-8 hours of focused work
**Impact**: 10x better usability!

---

## 🔧 Dependencies Check

Make sure you have:
```bash
pip install PyQt6 pyqtgraph numpy scipy
```

---

## 📞 Next Steps

1. **Review strategy document**: `EEG_GUI_UPGRADE_STRATEGY.md`
2. **Start with Quick Win #1**: Persistent curves (biggest impact)
3. **Test incrementally**: Verify each change works
4. **Commit often**: Save your progress

Ready to start coding? Let's do Quick Win #1 first! 🚀
