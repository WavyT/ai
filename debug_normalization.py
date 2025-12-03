#!/usr/bin/env python3
"""
Debug with normalization enabled (z-score).
"""

import numpy as np
from eeg_loader import EEGLoader

print("=" * 70)
print("TESTING WITH NORMALIZATION (z-score)")
print("=" * 70)

# Load test data
loader = EEGLoader("continuous.dat", num_channels=72)

# Load 2 channels
print("\n1. Loading channels 12 and 67...")
data = loader.load_channels([12, 67], start_sample=0, end_sample=10000, dtype=np.float32)
print(f"✓ Original data shape: {data.shape}")
print(f"   Ch 12 range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
print(f"   Ch 67 range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")

# Apply DC removal
print("\n2. Applying DC removal...")
data = data - data.mean(axis=0, keepdims=True)
print(f"   After DC removal:")
print(f"   Ch 12 range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
print(f"   Ch 67 range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")

# Apply normalization (z-score)
print("\n3. Applying normalization (z-score)...")
std = data.std(axis=0, keepdims=True)
std[std == 0] = 1  # Avoid division by zero
data = data / std

print(f"   After normalization:")
print(f"   Ch 12 range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
print(f"   Ch 67 range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")

# Now test channel spacing calculation
print("\n4. Testing channel spacing with normalized data...")

plot_data = data.copy()
base_y_scale = 7.99  # From screenshot

data_std = np.std(plot_data)
data_range = np.max(plot_data) - np.min(plot_data)

print(f"   Data std: {data_std:.2f}")
print(f"   Data range: {data_range:.2f}")
print(f"   Y-scale: {base_y_scale:.2f}")

# Base channel spacing depends on whether data is normalized or raw
if data_range < 10:
    base_spacing = max(data_std * 6.0, 5.0)
    print(f"   Using normalized formula: max(std * 6.0, 5.0)")
else:
    base_spacing = max(data_std * 4.0, data_range * 1.5)
    print(f"   Using raw data formula")

channel_spacing = base_spacing / base_y_scale

print(f"   Base spacing: {base_spacing:.2f}")
print(f"   Final channel spacing: {channel_spacing:.2f}")

# Simulate plotting
print("\n5. Simulating channel plotting...")
for i in range(2):
    channel_data = plot_data[:, i]
    channel_mean = channel_data.mean()
    channel_data_centered = channel_data - channel_mean
    y_scale = base_y_scale  # Apply same Y-scale
    scaled_data = channel_data_centered * y_scale
    y_data = scaled_data + (i * channel_spacing)

    print(f"\n   Channel {i} (actual ch {[12, 67][i]}):")
    print(f"     After centering: [{channel_data_centered.min():.2f}, {channel_data_centered.max():.2f}]")
    print(f"     After Y-scale ({y_scale:.2f}x): [{scaled_data.min():.2f}, {scaled_data.max():.2f}]")
    print(f"     After offset (i={i}): [{y_data.min():.2f}, {y_data.max():.2f}]")
    print(f"     Mean position: {y_data.mean():.2f}")

# Check overlap
print("\n6. Checking for overlap with normalized data...")
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

ch0_pp = ch0_max - ch0_min
ch1_pp = ch1_max - ch1_min

overlap = not (ch0_max < ch1_min or ch1_max < ch0_min)
separation = abs(ch1_final.mean() - ch0_final.mean())

print(f"   Ch 0: range=[{ch0_min:.2f}, {ch0_max:.2f}], mean={ch0_final.mean():.2f}, pk-pk={ch0_pp:.2f}")
print(f"   Ch 1: range=[{ch1_min:.2f}, {ch1_max:.2f}], mean={ch1_final.mean():.2f}, pk-pk={ch1_pp:.2f}")
print(f"   Separation between means: {separation:.2f}")
print(f"   Required spacing (max pk-pk): {max(ch0_pp, ch1_pp):.2f}")
print(f"   Current spacing: {channel_spacing:.2f}")
print(f"   Overlapping: {'YES ✗' if overlap else 'NO ✓'}")

if overlap or channel_spacing < max(ch0_pp, ch1_pp):
    print("\n   ✗✗✗ PROBLEM IDENTIFIED! ✗✗✗")
    print("   The channel spacing is TOO SMALL!")
    print(f"   Need at least: {max(ch0_pp, ch1_pp):.2f}")
    print(f"   Currently have: {channel_spacing:.2f}")
    print(f"   Missing: {max(ch0_pp, ch1_pp) - channel_spacing:.2f}")

print("\n" + "=" * 70)
