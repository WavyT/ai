#!/usr/bin/env python3
"""
Generate synthetic EEG test data for debugging the GUI.

Creates a continuous.dat file with:
- 72 channels
- Synthetic EEG-like signals
- Trigger channel with pulses (channel 67)
- Known artifacts for testing bad channel detection
"""

import numpy as np
import struct

def generate_synthetic_eeg(duration_seconds=300, sampling_rate=200, num_channels=72):
    """
    Generate synthetic EEG data.

    Args:
        duration_seconds: Duration in seconds
        sampling_rate: Sampling rate in Hz
        num_channels: Number of channels (default 72)

    Returns:
        data: numpy array of shape (num_samples, num_channels)
    """
    num_samples = int(duration_seconds * sampling_rate)
    data = np.zeros((num_samples, num_channels), dtype=np.float32)

    time = np.arange(num_samples) / sampling_rate

    print(f"Generating {duration_seconds}s of data ({num_samples} samples) for {num_channels} channels...")

    # Generate different signal types for different channels
    for ch in range(num_channels):
        if ch == 67:
            # Channel 67: Trigger channel with pulses
            # Create trigger pulses every 30 seconds (refractory period > 21s)
            trigger_times = np.arange(10, duration_seconds, 30)  # Triggers at 10s, 40s, 70s, etc.
            trigger_signal = np.zeros(num_samples)
            for t in trigger_times:
                # Create a 100ms pulse
                start_idx = int(t * sampling_rate)
                end_idx = int((t + 0.1) * sampling_rate)
                if end_idx < num_samples:
                    trigger_signal[start_idx:end_idx] = 2.0  # Pulse amplitude of 2.0
            data[:, ch] = trigger_signal
            print(f"  Ch {ch}: Trigger channel with {len(trigger_times)} pulses")

        elif ch < 10:
            # First 10 channels: Alpha band oscillations (8-12 Hz)
            freq = 10.0 + ch * 0.5
            amplitude = 50.0 + ch * 10
            data[:, ch] = amplitude * np.sin(2 * np.pi * freq * time)
            data[:, ch] += np.random.normal(0, 5, num_samples)  # Add noise
            print(f"  Ch {ch}: Alpha {freq:.1f} Hz, amp {amplitude:.1f}")

        elif ch < 20:
            # Next 10 channels: Beta band (13-30 Hz)
            freq = 15.0 + (ch - 10) * 1.5
            amplitude = 30.0 + (ch - 10) * 5
            data[:, ch] = amplitude * np.sin(2 * np.pi * freq * time)
            data[:, ch] += np.random.normal(0, 8, num_samples)

        elif ch < 30:
            # Next 10: Theta band (4-7 Hz)
            freq = 5.0 + (ch - 20) * 0.3
            amplitude = 60.0 + (ch - 20) * 8
            data[:, ch] = amplitude * np.sin(2 * np.pi * freq * time)
            data[:, ch] += np.random.normal(0, 10, num_samples)

        elif ch < 40:
            # Next 10: Delta band (1-4 Hz)
            freq = 2.0 + (ch - 30) * 0.2
            amplitude = 80.0 + (ch - 30) * 10
            data[:, ch] = amplitude * np.sin(2 * np.pi * freq * time)
            data[:, ch] += np.random.normal(0, 15, num_samples)

        elif ch == 50:
            # Channel 50: Flat line (bad channel)
            data[:, ch] = 0.0
            print(f"  Ch {ch}: FLAT LINE (bad channel)")

        elif ch == 51:
            # Channel 51: Excessive noise (bad channel)
            data[:, ch] = np.random.normal(0, 500, num_samples)
            print(f"  Ch {ch}: EXCESSIVE NOISE (bad channel)")

        elif ch == 52:
            # Channel 52: 60 Hz powerline noise
            data[:, ch] = 40 * np.sin(2 * np.pi * 8 * time)  # Base signal
            data[:, ch] += 100 * np.sin(2 * np.pi * 60 * time)  # Strong 60 Hz
            print(f"  Ch {ch}: 60 Hz powerline noise")

        elif ch == 53:
            # Channel 53: Spike artifacts
            base_signal = 50 * np.sin(2 * np.pi * 10 * time)
            spike_times = np.random.choice(num_samples, size=100, replace=False)
            for spike_idx in spike_times:
                if spike_idx + 10 < num_samples:
                    base_signal[spike_idx:spike_idx+10] += 500  # Large spike
            data[:, ch] = base_signal
            print(f"  Ch {ch}: Spike artifacts")

        else:
            # Rest: Mix of frequencies
            freq1 = 5.0 + (ch % 10) * 0.8
            freq2 = 12.0 + (ch % 8) * 1.2
            amplitude = 40.0 + (ch % 15) * 5
            data[:, ch] = amplitude * (
                np.sin(2 * np.pi * freq1 * time) +
                0.5 * np.sin(2 * np.pi * freq2 * time)
            )
            data[:, ch] += np.random.normal(0, 8, num_samples)

    print(f"\nData shape: {data.shape}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

    return data

def write_continuous_dat(data, filename="continuous.dat"):
    """
    Write data in the format expected by EEGLoader.

    Format: interleaved samples (sample0_ch0, sample0_ch1, ..., sample0_ch71,
                                  sample1_ch0, sample1_ch1, ..., sample1_ch71, ...)
    """
    num_samples, num_channels = data.shape

    print(f"\nWriting {filename}...")
    print(f"  Format: int16, interleaved by sample")
    print(f"  Total values: {num_samples * num_channels:,}")

    # Convert to int16 (typical for raw EEG data)
    # Scale to use reasonable int16 range
    data_scaled = data.astype(np.float32)

    # Convert to int16
    data_int16 = data_scaled.astype(np.int16)

    # Write interleaved format
    with open(filename, 'wb') as f:
        for sample_idx in range(num_samples):
            for ch_idx in range(num_channels):
                value = data_int16[sample_idx, ch_idx]
                f.write(struct.pack('<h', value))  # little-endian int16

    file_size_mb = (num_samples * num_channels * 2) / (1024 * 1024)
    print(f"  Written: {file_size_mb:.2f} MB")
    print(f"  ✓ {filename} created successfully!")

def create_timestamps(num_samples, sampling_rate=200):
    """Create timestamps file."""
    timestamps = np.arange(num_samples) / sampling_rate

    filename = "timestamps.npy"
    np.save(filename, timestamps.astype(np.float64))
    print(f"\n✓ {filename} created ({len(timestamps)} timestamps)")

def main():
    """Generate test data."""
    print("=" * 60)
    print("EEG Test Data Generator")
    print("=" * 60)

    # Generate 5 minutes of data (fast to generate, enough to test)
    duration_seconds = 300  # 5 minutes
    sampling_rate = 200
    num_channels = 72

    # Generate data
    data = generate_synthetic_eeg(
        duration_seconds=duration_seconds,
        sampling_rate=sampling_rate,
        num_channels=num_channels
    )

    # Write files
    write_continuous_dat(data, "continuous.dat")
    create_timestamps(data.shape[0], sampling_rate)

    print("\n" + "=" * 60)
    print("Test data generation complete!")
    print("=" * 60)
    print("\nTo test the GUI:")
    print("  python3 advanced_eeg_gui2.py")
    print("\nTest features:")
    print("  • Trigger channel: Ch 67 (pulses every 30s)")
    print("  • Bad channels: Ch 50 (flat), Ch 51 (noise)")
    print("  • Artifacts: Ch 52 (60Hz), Ch 53 (spikes)")
    print("  • Normal signals: Ch 0-49, 54-66, 68-71")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
