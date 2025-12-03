#!/usr/bin/env python3
"""
Debug script to analyze the issues without running full GUI.
"""

import numpy as np
from eeg_loader import EEGLoader

print("=" * 70)
print("DEBUGGING EEG GUI ISSUES")
print("=" * 70)

# Load test data
print("\n1. Loading test data...")
loader = EEGLoader("continuous.dat", num_channels=72)
print(f"✓ Loaded {loader.num_channels} channels, {loader.num_samples_per_channel:,} samples")

# Load 2 channels
print("\n2. Loading channels 12 and 67...")
data = loader.load_channels([12, 67], start_sample=0, end_sample=10000, dtype=np.float32)
print(f"✓ Data shape: {data.shape}")
print(f"   Ch 12 range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
print(f"   Ch 67 range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")

# Simulate channel spacing calculation (from GUI code lines 2001-2022)
print("\n3. Testing channel spacing calculation...")

plot_data = data.copy()
base_y_scale = 1.0

# Calculate base spacing from data statistics
data_std = np.std(plot_data)
data_range = np.max(plot_data) - np.min(plot_data)

print(f"   Data std: {data_std:.2f}")
print(f"   Data range: {data_range:.2f}")

# Base channel spacing depends on whether data is normalized or raw
if data_range < 10:
    # Normalized/small data: use std-based spacing
    base_spacing = max(data_std * 6.0, 5.0)
    print(f"   Using normalized formula: std * 6.0")
else:
    # Raw data: use range-based spacing
    base_spacing = max(data_std * 4.0, data_range * 1.5)
    print(f"   Using raw data formula: max(std * 4.0, range * 1.5)")

channel_spacing = base_spacing / base_y_scale

print(f"   Base spacing: {base_spacing:.2f}")
print(f"   Final channel spacing: {channel_spacing:.2f}")

# Simulate plotting transformation
print("\n4. Simulating channel offset calculation...")
for i in range(2):
    channel_data = plot_data[:, i]
    channel_mean = channel_data.mean()
    channel_data_centered = channel_data - channel_mean
    y_scale = 1.0
    scaled_data = channel_data_centered * y_scale
    y_data = scaled_data + (i * channel_spacing)

    print(f"\n   Channel {i} (actual ch {[12, 67][i]}):")
    print(f"     Original range: [{channel_data.min():.2f}, {channel_data.max():.2f}]")
    print(f"     After centering: [{channel_data_centered.min():.2f}, {channel_data_centered.max():.2f}]")
    print(f"     After offset (i={i}): [{y_data.min():.2f}, {y_data.max():.2f}]")
    print(f"     Mean position: {y_data.mean():.2f}")

# Check if channels overlap
print("\n5. Checking for overlap...")
ch0_data = plot_data[:, 0]
ch1_data = plot_data[:, 1]

ch0_centered = ch0_data - ch0_data.mean()
ch1_centered = ch1_data - ch1_data.mean()

ch0_final = ch0_centered + (0 * channel_spacing)
ch1_final = ch1_centered + (1 * channel_spacing)

ch0_min, ch0_max = ch0_final.min(), ch0_final.max()
ch1_min, ch1_max = ch1_final.min(), ch1_final.max()

overlap = not (ch0_max < ch1_min or ch1_max < ch0_min)
separation = abs(ch1_final.mean() - ch0_final.mean())

print(f"   Ch 0: range=[{ch0_min:.2f}, {ch0_max:.2f}], mean={ch0_final.mean():.2f}")
print(f"   Ch 1: range=[{ch1_min:.2f}, {ch1_max:.2f}], mean={ch1_final.mean():.2f}")
print(f"   Separation between means: {separation:.2f}")
print(f"   Expected spacing: {channel_spacing:.2f}")
print(f"   Overlapping: {'YES ✗' if overlap else 'NO ✓'}")

# Identify the problem
print("\n6. PROBLEM ANALYSIS:")
if overlap:
    print("   ✗ Channels ARE overlapping!")
    print(f"   ✗ Ch 0 peak-to-peak: {ch0_max - ch0_min:.2f}")
    print(f"   ✗ Ch 1 peak-to-peak: {ch1_max - ch1_min:.2f}")
    print(f"   ✗ Required spacing for no overlap: {max(ch0_max - ch0_min, ch1_max - ch1_min):.2f}")
    print(f"   ✗ Current spacing: {channel_spacing:.2f}")

    if channel_spacing < max(ch0_max - ch0_min, ch1_max - ch1_min):
        print("\n   ROOT CAUSE: Channel spacing is smaller than peak-to-peak amplitude!")
        print("   SOLUTION: Calculate spacing based on PER-CHANNEL peak-to-peak, not global stats")
else:
    print("   ✓ Channels are properly separated!")

# Test trigger detection
print("\n7. Testing trigger detection...")
from advanced_eeg_gui2 import TriggerDetector

trigger_data = loader.load_channels([67], start_sample=0, end_sample=60000, dtype=np.float32)[:, 0]
print(f"   Trigger channel range: [{trigger_data.min():.2f}, {trigger_data.max():.2f}]")

detector = TriggerDetector(threshold=1.2, refractory_seconds=21.0, sampling_rate=200.0)
triggers = detector.detect(trigger_data, start_sample=0)

print(f"   Triggers detected: {len(triggers)}")
if len(triggers) > 0:
    trigger_times = [t / 200.0 for t in triggers]
    print(f"   First 5 trigger times: {trigger_times[:5]}")
    print(f"   Expected: triggers every 30s starting at 10s")
    print(f"   Actual intervals: {[trigger_times[i+1] - trigger_times[i] for i in range(min(4, len(trigger_times)-1))]}")
else:
    print(f"   ✗ NO TRIGGERS DETECTED!")
    print(f"   Threshold: 1.2, Max value in trigger channel: {trigger_data.max():.2f}")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
