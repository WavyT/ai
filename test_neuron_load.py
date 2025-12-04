#!/usr/bin/env python3
"""
Test script for neuron.py EEG loading functionality
Tests loading of .dat and .mat files without GUI
"""

import numpy as np
from pathlib import Path
import sys

# Try importing necessary modules
try:
    from eeg_loader import EEGLoader
    print("✓ eeg_loader.py imported successfully")
except ImportError as e:
    print(f"✗ Failed to import eeg_loader: {e}")
    sys.exit(1)

try:
    import scipy.io as sio
    print("✓ scipy imported successfully")
except ImportError:
    print("✗ scipy not available - .mat file loading will not work")
    print("  Install with: pip install scipy")

print("\n" + "="*60)
print("Testing EEG File Loading")
print("="*60)

# Look for test files
current_dir = Path('.')
dat_files = list(current_dir.glob('*.dat'))
mat_files = list(current_dir.glob('*4AP*.mat')) + list(current_dir.glob('*Illumination*.mat'))

print(f"\nFound {len(dat_files)} .dat files:")
for f in dat_files:
    print(f"  - {f.name}")

print(f"\nFound {len(mat_files)} matching .mat files:")
for f in mat_files:
    print(f"  - {f.name}")

# Test loading a .dat file if available
if dat_files:
    print("\n" + "-"*60)
    print("Testing .dat file loading")
    print("-"*60)

    dat_file = dat_files[0]
    print(f"\nLoading: {dat_file}")

    try:
        loader = EEGLoader(str(dat_file))
        print(f"✓ Successfully initialized loader")
        print(f"  Channels: {loader.num_channels}")
        print(f"  Samples per channel: {loader.num_samples_per_channel:,}")

        # Load a small chunk
        print("\nLoading first 1000 samples...")
        data = loader.load_all_channels(start_sample=0, end_sample=1000, dtype=np.float32)
        print(f"✓ Loaded data shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Data mean: {data.mean():.2f}")

    except Exception as e:
        print(f"✗ Failed to load .dat file: {e}")

# Test loading a .mat file if available
if mat_files:
    print("\n" + "-"*60)
    print("Testing .mat file loading")
    print("-"*60)

    mat_file = mat_files[0]
    print(f"\nLoading: {mat_file}")

    try:
        mat_data = sio.loadmat(str(mat_file))
        print(f"✓ Successfully loaded .mat file")
        print(f"\nKeys in MAT file:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                val = mat_data[key]
                if isinstance(val, np.ndarray):
                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"  {key}: {type(val)}")

        # Try to extract data
        if 'Meting' in mat_data:
            print("\n✓ Found 'Meting' structure (Neuron.m format)")
            meting = mat_data['Meting']
            if hasattr(meting, 'dtype') and meting.dtype.names:
                print(f"  Fields: {list(meting.dtype.names)}")
                if 'adc' in meting.dtype.names:
                    adc_data = meting['adc'][0, 0]
                    print(f"  ADC data shape: {adc_data.shape}")

        elif 'data' in mat_data:
            print("\n✓ Found 'data' array")
            data = mat_data['data']
            print(f"  Shape: {data.shape}")
            print(f"  Type: {data.dtype}")
            print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")

        elif 'adc' in mat_data:
            print("\n✓ Found 'adc' array")
            data = mat_data['adc']
            print(f"  Shape: {data.shape}")
            print(f"  Type: {data.dtype}")

        else:
            print("\n⚠ No standard data keys found, trying largest array...")
            largest_key = None
            largest_size = 0
            for key in mat_data.keys():
                if not key.startswith('__'):
                    arr = mat_data[key]
                    if isinstance(arr, np.ndarray) and arr.size > largest_size:
                        largest_key = key
                        largest_size = arr.size

            if largest_key:
                print(f"  Largest array: '{largest_key}'")
                data = mat_data[largest_key]
                print(f"  Shape: {data.shape}")
                print(f"  Type: {data.dtype}")

    except Exception as e:
        print(f"✗ Failed to load .mat file: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Test complete!")
print("="*60)

if dat_files or mat_files:
    print("\nYou can now run neuron.py and load these files through the GUI:")
    print("  python neuron.py")
else:
    print("\n⚠ No test files found in current directory")
    print("  Place .dat or .mat files here to test loading")
