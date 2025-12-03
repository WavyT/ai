#!/usr/bin/env python3
"""
Test the channel spacing fix.
"""

import numpy as np

# Simulate the fix
plot_data = np.array([
    [-2.39, -0.06],  # Min values after normalization
    [2.31, 15.78]    # Max values after normalization
]).T  # Shape: (2, 2) but we'll use proper data

# Load real data
from eeg_loader import EEGLoader
loader = EEGLoader("continuous.dat", num_channels=72)
data = loader.load_channels([12, 67], start_sample=0, end_sample=10000, dtype=np.float32)

# Apply processing
data = data - data.mean(axis=0, keepdims=True)  # DC removal
std = data.std(axis=0, keepdims=True)
std[std == 0] = 1
data = data / std  # Normalization

plot_data = data
selected_channels = [12, 67]
base_y_scale = 7.99
channel_y_scales = {}

print("=" * 70)
print("TESTING CHANNEL SPACING FIX")
print("=" * 70)

# NEW SPACING CALCULATION (from fix)
print("\n1. New spacing calculation (per-channel peak-to-peak)...")
max_pp_range = 0.0
for i in range(len(selected_channels)):
    channel_data = plot_data[:, i]
    channel_mean = channel_data.mean()
    channel_data_centered = channel_data - channel_mean

    # Apply Y-scale
    ch_idx = selected_channels[i]
    y_scale = channel_y_scales.get(ch_idx, base_y_scale)
    scaled_data = channel_data_centered * y_scale

    # Get peak-to-peak
    pp_range = scaled_data.max() - scaled_data.min()
    max_pp_range = max(max_pp_range, pp_range)

    print(f"   Ch {ch_idx}: pk-pk = {pp_range:.2f}")

channel_spacing = max_pp_range * 1.2
channel_spacing = max(channel_spacing, 10.0)

print(f"\n   Max pk-pk: {max_pp_range:.2f}")
print(f"   With 20% padding: {channel_spacing:.2f}")

# Test overlap
print("\n2. Testing for overlap with new spacing...")
ch0_data = plot_data[:, 0]
ch1_data = plot_data[:, 1]

ch0_centered = ch0_data - ch0_data.mean()
ch1_centered = ch1_data - ch1_data.mean()

ch0_scaled = ch0_centered * base_y_scale
ch1_scaled = ch1_centered * base_y_scale

ch0_final = ch0_scaled + (0 * channel_spacing)
ch1_final = ch1_scaled + (1 * channel_spacing)

ch0_min, ch0_max = ch0_final.min(), ch0_final.max()
ch1_min, ch1_max = ch1_final.min(), ch1_final.max()

overlap = not (ch0_max < ch1_min or ch1_max < ch0_min)

print(f"   Ch 0: range=[{ch0_min:.2f}, {ch0_max:.2f}]")
print(f"   Ch 1: range=[{ch1_min:.2f}, {ch1_max:.2f}]")
print(f"   Spacing: {channel_spacing:.2f}")
print(f"   Overlapping: {'YES ✗' if overlap else 'NO ✓✓✓'}")

if not overlap:
    print("\n   ✓✓✓ FIX SUCCESSFUL! Channels no longer overlap! ✓✓✓")
else:
    print("\n   ✗ Still overlapping")

print("\n" + "=" * 70)
