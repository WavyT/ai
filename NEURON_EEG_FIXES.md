# Neuron.py EEG Data Loading Fixes

## Summary

Fixed `neuron.py` to properly load and display multichannel EEG data from both `.dat` and `.mat` files, matching the behavior of the original MATLAB `Neuron.m` application.

## What Was Fixed

### 1. **EEG Loader Integration**
- Integrated the existing `eeg_loader.py` module for `.dat` file loading
- Added `load_dat_file()` method that uses the `EEGLoader` class
- Automatically detects channel count or tries common values (16, 32, 64, 72, 128)
- Loads data in chunks (100k samples max for initial display)

### 2. **MAT File Loading**
Completely rewrote `load_mat_file()` to handle multiple formats:

- **Neuron.m Meting structures**: Extracts `adc`, `dac`, and metadata from MATLAB structures
- **Raw EEG arrays**: Loads data from keys like `'data'`, `'adc'`, etc.
- **Auto-detection**: Finds the largest array if no standard keys exist
- **Shape handling**: Properly handles 1D, 2D, and 3D data arrays

### 3. **EEG Visualization**
Rewrote the `redraw()` function for proper EEG display:

- **Multi-channel stacked view**: Similar to standard EEG viewers
- **Single channel mode**: When a specific channel is selected
- **Adjustable gain**: Use the gain spinbox to scale traces
- **Channel separation**: Automatic spacing between channels
- **Color coding**: Each channel has a distinct color
- **Channel labels**: Labels displayed on the left side

## File Format Support

### `.dat` Files (Continuous EEG)
```
Format: Interleaved int16 multichannel data
Structure: [ch1_s1, ch2_s1, ..., chN_s1, ch1_s2, ch2_s2, ...]
Loader: Uses EEGLoader class from eeg_loader.py
Output shape: (samples, channels)
```

### `.mat` Files (MATLAB)
```
Supported structures:
1. Meting structure (Neuron.m format)
   - Contains: adc, dac, ADC, DAC, metadata

2. Raw data arrays
   - Key: 'data', 'adc', or any large array
   - Auto-converted to float32

3. 3D arrays (samples x channels x sweeps)
   - Automatically extracts first sweep
```

### `.h5` Files (MEA/HDF5)
```
Support for HDF5-based MEA data files
```

## Usage

### Running the Application

```bash
python neuron.py
```

### Loading Data

1. Click **"Load Data"** button in the Command tab
2. Select file type:
   - `.dat` files (continuous EEG)
   - `.mat` files (MATLAB/Neuron format)
   - `.h5` files (MEA data)
3. The file will be loaded and displayed automatically

### Viewing Options

**Multi-channel view (default):**
- Shows up to 16 channels stacked vertically
- Each channel is color-coded
- Channel labels on the left

**Single channel view:**
- Set "Channel" spinbox to a specific channel number (1-N)
- Shows only that channel with full details

**Gain adjustment:**
- Use "Gain" spinbox to scale the traces
- Default: 1.0x
- Range: 0.1x to 1000x

## Testing

### Quick Test
```bash
# Test file loading without GUI
python test_neuron_load.py
```

This will:
- Check for `eeg_loader.py` and dependencies
- Find `.dat` and `.mat` files in the current directory
- Test loading and display basic statistics

### Manual Testing
1. Place `4AP+Illumination.mat` or `continuous.dat` in the same directory
2. Run `python neuron.py`
3. Load the file
4. Verify multichannel traces are displayed

## Known Limitations

1. **Large files**: Only loads first 100,000 samples initially (can be extended)
2. **Channel limit**: Displays maximum 16 channels at once in stacked view
3. **Memory**: Large multichannel files are loaded into RAM (consider chunking for very large files)

## Dependencies

Required:
- `numpy`
- `scipy` (for .mat files)
- `matplotlib`
- `PyQt5`
- `h5py` (for .h5 files)

```bash
pip install numpy scipy matplotlib PyQt5 h5py
```

## File Structure

```
neuron.py           # Main application (fixed)
eeg_loader.py       # EEG data loader for .dat files
test_neuron_load.py # Test script for loading
```

## Implementation Details

### Data Flow

```
.dat file → EEGLoader → load_all_channels() → self.meting.adc (samples, channels)
                                                      ↓
                                                  redraw()
                                                      ↓
                                            matplotlib display
```

```
.mat file → scipy.io.loadmat() → extract arrays → self.meting.adc (samples, channels)
                                                         ↓
                                                     redraw()
                                                         ↓
                                                matplotlib display
```

### Key Functions

**`load_dat_file(filename)`**
- Uses EEGLoader to read interleaved int16 data
- Auto-detects or tries common channel counts
- Creates ADC channel metadata
- Converts to float32 for display

**`load_mat_file(filename)`**
- Loads MAT file with scipy.io.loadmat()
- Checks for standard keys: 'Meting', 'data', 'adc'
- Falls back to largest array if no standard keys
- Ensures 2D shape (samples x channels)
- Creates channel metadata if missing

**`redraw()`**
- Handles both single and multi-channel display
- Stacked view with offset and color coding
- Respects gain and channel selection settings
- Adds channel labels on left side

## Troubleshooting

### "No data displayed after loading"
**Cause**: Data shape is not recognized or is empty
**Solution**: Check the console output for error messages. Verify file contains data.

### "Channel count detection failed"
**Cause**: File size doesn't divide evenly by common channel counts
**Solution**: Manually specify channel count in the code or create a `channel_info.txt` file

### "scipy not installed"
**Cause**: scipy is required for .mat files
**Solution**: `pip install scipy`

### "EEGLoader not found"
**Cause**: `eeg_loader.py` is not in the same directory
**Solution**: Ensure `eeg_loader.py` is in the same folder as `neuron.py`

## Future Improvements

1. **Chunked loading**: Stream large files instead of loading all at once
2. **Channel selection dialog**: Allow user to select which channels to display
3. **Time axis**: Convert samples to time (requires sampling rate)
4. **Zoom and pan**: Interactive navigation of long recordings
5. **Filtering**: Real-time filters (highpass, lowpass, notch)
6. **Annotations**: Mark events and artifacts
7. **Export**: Save selected segments or filtered data

## Commit Information

**Branch**: `claude/matlab-to-python-neuron-01Y7dxCdr7CcLT8sRSF4vejn`

**Commit**: "Fix neuron.py to load and display multichannel EEG data"

Changes:
- Added EEGLoader integration
- Rewrote load_mat_file() with auto-detection
- Rewrote redraw() for EEG-style visualization
- Added support for .dat files
- Improved error handling and user feedback
