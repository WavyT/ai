"""
EEG Data Loader for continuous.dat files
Based on the MATLAB readmulti_frank.m function structure.

The file contains multi-channel EEG data in interleaved format:
- Data type: int16 (2 bytes per sample)
- Structure: [ch1_sample1, ch2_sample1, ..., chN_sample1, ch1_sample2, ...]
- Channels are interleaved, then samples are sequential

Memory-efficient loading strategies:
- Memory-mapped files for random access
- Chunked reading for sequential processing
- Selective channel loading
"""

import numpy as np
import os
from pathlib import Path
from typing import Union, List, Optional, Tuple
import warnings


class EEGLoader:
    """
    Efficient loader for large multi-channel EEG .dat files.
    
    Based on Buzsaki lab format and readmulti_frank.m structure.
    """
    
    def __init__(self, filepath: Union[str, Path], num_channels: Optional[int] = None):
        """
        Initialize EEG loader.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the continuous.dat file
        num_channels : int, optional
            Number of channels in the file. If None, will be inferred from file size
            and available metadata (sample_numbers.npy).
        """
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"EEG file not found: {filepath}")
        
        # File metadata
        self.file_size = self.filepath.stat().st_size
        self.dtype = np.int16
        self.bytes_per_sample = np.dtype(self.dtype).itemsize  # 2 bytes for int16
        
        # Try to load metadata files if they exist
        self.metadata_dir = self.filepath.parent
        self.sample_numbers = None
        self.timestamps = None
        
        self._load_metadata()
        
        # Infer or set number of channels
        if num_channels is None:
            self.num_channels = self._infer_num_channels()
        else:
            self.num_channels = num_channels
            self._verify_channel_count()
        
        # Calculate number of samples per channel
        self.num_samples_per_channel = self.file_size // (self.bytes_per_sample * self.num_channels)
        
        # Actual file size that will be used (rounded down to complete frames)
        self.actual_file_size = self.num_samples_per_channel * self.bytes_per_sample * self.num_channels
        
        print(f"EEG File Information:")
        print(f"  File: {self.filepath.name}")
        print(f"  File size: {self.file_size / (1024**3):.3f} GB")
        print(f"  Number of channels: {self.num_channels}")
        print(f"  Samples per channel: {self.num_samples_per_channel:,}")
        
        # Calculate duration if timestamps are available
        if self.timestamps is not None and len(self.timestamps) > 1:
            duration_sec = self.timestamps[-1] - self.timestamps[0]
            sampling_rate = 1.0 / np.mean(np.diff(self.timestamps))
            print(f"  Sampling rate: {sampling_rate:.2f} Hz")
            print(f"  Total duration: {duration_sec/60:.2f} minutes ({duration_sec:.1f} seconds)")
        else:
            print(f"  Duration: Unknown (timestamps not available)")
        
    def _load_metadata(self):
        """Load metadata files if available (sample_numbers.npy, timestamps.npy)."""
        sample_file = self.metadata_dir / "sample_numbers.npy"
        timestamp_file = self.metadata_dir / "timestamps.npy"
        
        if sample_file.exists():
            try:
                self.sample_numbers = np.load(sample_file)
                print(f"  Loaded sample_numbers.npy: {len(self.sample_numbers):,} samples")
            except Exception as e:
                warnings.warn(f"Could not load sample_numbers.npy: {e}")
        
        if timestamp_file.exists():
            try:
                self.timestamps = np.load(timestamp_file)
                print(f"  Loaded timestamps.npy: {len(self.timestamps):,} timestamps")
            except Exception as e:
                warnings.warn(f"Could not load timestamps.npy: {e}")
        
        # Verify metadata consistency
        if self.sample_numbers is not None and self.timestamps is not None:
            if len(self.sample_numbers) != len(self.timestamps):
                warnings.warn(f"Mismatch: {len(self.sample_numbers)} samples vs {len(self.timestamps)} timestamps")
    
    def _infer_num_channels(self) -> int:
        """Infer number of channels from file size and metadata."""
        if self.sample_numbers is not None:
            num_samples = len(self.sample_numbers)
            num_channels = (self.file_size // self.bytes_per_sample) // num_samples
            if num_channels > 0:
                print(f"  Inferred {num_channels} channels from file size and metadata")
                return num_channels
        
        # If no metadata, try common channel counts
        common_channels = [16, 32, 64, 72, 128, 256]
        for nch in common_channels:
            samples = self.file_size // (self.bytes_per_sample * nch)
            remainder = self.file_size % (self.bytes_per_sample * nch)
            if remainder == 0:
                print(f"  Inferred {nch} channels (file divides evenly)")
                return nch
        
        # Default: calculate from file size assuming standard sampling
        # This is less reliable, so we'll raise a warning
        estimated_channels = (self.file_size // self.bytes_per_sample) // 1000000
        warnings.warn(f"Could not reliably infer channel count. Estimated: {estimated_channels}")
        return estimated_channels
    
    def _verify_channel_count(self):
        """Verify that the provided channel count is consistent with file size."""
        expected_samples = self.file_size // (self.bytes_per_sample * self.num_channels)
        remainder = self.file_size % (self.bytes_per_sample * self.num_channels)
        
        if remainder != 0:
            warnings.warn(
                f"File size ({self.file_size} bytes) is not a multiple of "
                f"{self.num_channels} channels * {self.bytes_per_sample} bytes. "
                f"Last {remainder} bytes will be ignored."
            )
    
    def probe_structure(self, num_samples: int = 100) -> dict:
        """
        Probe the file structure by reading a small sample.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to read for probing
            
        Returns:
        --------
        dict : Dictionary with file structure information
        """
        # Read first chunk
        with open(self.filepath, 'rb') as f:
            # Read first num_samples worth of data
            chunk_size = num_samples * self.num_channels * self.bytes_per_sample
            data_raw = f.read(chunk_size)
            data = np.frombuffer(data_raw, dtype=self.dtype)
            
            # Reshape to [channels, samples]
            data_reshaped = data.reshape(self.num_channels, num_samples)
            
            # Statistics per channel
            channel_stats = {
                'mean': np.mean(data_reshaped, axis=1),
                'std': np.std(data_reshaped, axis=1),
                'min': np.min(data_reshaped, axis=1),
                'max': np.max(data_reshaped, axis=1),
                'range': np.ptp(data_reshaped, axis=1)
            }
            
            # Read last chunk for comparison
            f.seek(-chunk_size, 2)  # Seek from end
            data_raw_end = f.read(chunk_size)
            data_end = np.frombuffer(data_raw_end, dtype=self.dtype)
            data_reshaped_end = data_end.reshape(self.num_channels, num_samples)
            
        return {
            'first_samples': data_reshaped[:, :min(10, num_samples)],
            'last_samples': data_reshaped_end[:, :min(10, num_samples)],
            'channel_statistics': channel_stats,
            'data_type': str(self.dtype),
            'num_channels': self.num_channels,
            'bytes_per_sample': self.bytes_per_sample,
            'samples_read': num_samples
        }
    
    def load_all_channels(self, 
                         start_sample: int = 0, 
                         end_sample: Optional[int] = None,
                         dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Load all channels from the file (memory-efficient for large files).
        
        Parameters:
        -----------
        start_sample : int
            Starting sample index (0-indexed)
        end_sample : int, optional
            Ending sample index (exclusive). If None, loads until end.
        dtype : np.dtype, optional
            Output data type. If None, uses int16.
            
        Returns:
        --------
        np.ndarray
            Shape: (num_samples, num_channels)
        """
        if end_sample is None:
            end_sample = self.num_samples_per_channel
        
        if start_sample < 0:
            start_sample = max(0, self.num_samples_per_channel + start_sample)
        
        if end_sample > self.num_samples_per_channel:
            end_sample = self.num_samples_per_channel
        
        num_samples = end_sample - start_sample
        
        if num_samples <= 0:
            raise ValueError(f"Invalid sample range: {start_sample} to {end_sample}")
        
        # Use memory-mapped file for efficiency
        memmap = np.memmap(
            self.filepath,
            dtype=self.dtype,
            mode='r',
            shape=(self.num_samples_per_channel, self.num_channels),
            order='C'
        )
        
        # Extract the requested range
        data = memmap[start_sample:end_sample, :].copy()
        
        # Convert dtype if requested
        if dtype is not None and dtype != self.dtype:
            data = data.astype(dtype)
        
        return data
    
    def load_channels(self,
                     channel_indices: Union[int, List[int]],
                     start_sample: int = 0,
                     end_sample: Optional[int] = None,
                     dtype: Optional[np.dtype] = None,
                     chunk_size: int = 4096) -> np.ndarray:
        """
        Load specific channels efficiently using chunked reading.
        
        Parameters:
        -----------
        channel_indices : int or list of int
            Channel index/indices to load (0-indexed)
        start_sample : int
            Starting sample index
        end_sample : int, optional
            Ending sample index. If None, loads until end.
        dtype : np.dtype, optional
            Output data type
        chunk_size : int
            Number of samples to read per chunk (for memory efficiency)
            
        Returns:
        --------
        np.ndarray
            Shape: (num_samples, num_selected_channels)
        """
        if isinstance(channel_indices, int):
            channel_indices = [channel_indices]
        
        channel_indices = np.array(channel_indices)
        
        # Validate channel indices
        if np.any(channel_indices < 0) or np.any(channel_indices >= self.num_channels):
            raise ValueError(f"Channel indices must be in range [0, {self.num_channels-1}]")
        
        if end_sample is None:
            end_sample = self.num_samples_per_channel
        
        if start_sample < 0:
            start_sample = max(0, self.num_samples_per_channel + start_sample)
        
        if end_sample > self.num_samples_per_channel:
            end_sample = self.num_samples_per_channel
        
        num_samples = end_sample - start_sample
        
        if num_samples <= 0:
            raise ValueError(f"Invalid sample range: {start_sample} to {end_sample}")
        
        # Pre-allocate output array
        output = np.zeros((num_samples, len(channel_indices)), dtype=dtype or self.dtype)
        
        # Calculate byte positions
        start_byte = start_sample * self.num_channels * self.bytes_per_sample
        end_byte = end_sample * self.num_channels * self.bytes_per_sample
        
        # Chunked reading
        with open(self.filepath, 'rb') as f:
            f.seek(start_byte)
            
            current_pos = 0
            while current_pos < num_samples:
                # Calculate chunk size for this iteration
                samples_to_read = min(chunk_size, num_samples - current_pos)
                bytes_to_read = samples_to_read * self.num_channels * self.bytes_per_sample
                
                # Read raw bytes
                raw_data = f.read(bytes_to_read)
                if len(raw_data) < bytes_to_read:
                    break  # End of file
                
                # Convert to numpy array
                data = np.frombuffer(raw_data, dtype=self.dtype)
                
                # Reshape: [samples, channels]
                data_reshaped = data.reshape(samples_to_read, self.num_channels)
                
                # Extract selected channels
                output[current_pos:current_pos + samples_to_read, :] = \
                    data_reshaped[:, channel_indices]
                
                current_pos += samples_to_read
        
        # Convert dtype if requested
        if dtype is not None and dtype != self.dtype:
            output = output.astype(dtype)
        
        return output
    
    def get_memory_map(self) -> np.memmap:
        """
        Get a memory-mapped view of the entire file.
        
        Returns:
        --------
        np.memmap
            Memory-mapped array of shape (num_samples, num_channels)
        """
        return np.memmap(
            self.filepath,
            dtype=self.dtype,
            mode='r',
            shape=(self.num_samples_per_channel, self.num_channels),
            order='C'
        )
    
    def get_sample_info(self) -> dict:
        """Get information about samples and timestamps."""
        info = {
            'num_samples_per_channel': self.num_samples_per_channel,
            'num_channels': self.num_channels,
            'file_size_bytes': self.file_size,
            'actual_used_bytes': self.actual_file_size,
            'has_sample_numbers': self.sample_numbers is not None,
            'has_timestamps': self.timestamps is not None
        }
        
        if self.sample_numbers is not None:
            info['sample_number_range'] = (self.sample_numbers.min(), self.sample_numbers.max())
            info['num_sample_numbers'] = len(self.sample_numbers)
        
        if self.timestamps is not None:
            info['timestamp_range'] = (self.timestamps.min(), self.timestamps.max())
            info['num_timestamps'] = len(self.timestamps)
            if len(self.timestamps) > 1:
                info['avg_sampling_period'] = np.mean(np.diff(self.timestamps))
        
        return info


def main():
    """Example usage and file structure probing."""
    # Find the continuous.dat file in the current directory
    dat_file = Path("continuous.dat")
    
    if not dat_file.exists():
        print(f"Error: {dat_file} not found in current directory.")
        print("Please run this script from the directory containing continuous.dat")
        return
    
    print("=" * 60)
    print("EEG File Structure Analysis")
    print("=" * 60)
    
    # Initialize loader - let it infer channels
    print("\n1. Initializing loader...")
    try:
        loader = EEGLoader(dat_file)
    except Exception as e:
        print(f"Error initializing loader: {e}")
        print("\nTrying with explicit channel count...")
        # Try with common channel counts
        for nch in [64, 72, 128]:
            try:
                loader = EEGLoader(dat_file, num_channels=nch)
                break
            except:
                continue
        else:
            print("Could not initialize loader. Please specify num_channels manually.")
            return
    
    # Probe file structure
    print("\n2. Probing file structure...")
    try:
        probe_result = loader.probe_structure(num_samples=1000)
        print(f"\n   First 5 samples (first 5 channels):")
        print(probe_result['first_samples'][:5, :5])
        print(f"\n   Channel statistics (first 5 channels):")
        stats = probe_result['channel_statistics']
        for ch in range(min(5, loader.num_channels)):
            print(f"   Channel {ch}: mean={stats['mean'][ch]:.1f}, "
                  f"std={stats['std'][ch]:.1f}, "
                  f"range=[{stats['min'][ch]}, {stats['max'][ch]}]")
    except Exception as e:
        print(f"   Warning: Could not probe structure: {e}")
    
    # Get sample info
    print("\n3. Sample information:")
    sample_info = loader.get_sample_info()
    for key, value in sample_info.items():
        print(f"   {key}: {value}")
    
    # Example: Load a small chunk from first channel
    print("\n4. Testing data loading...")
    try:
        print("   Loading first 1000 samples from channel 0...")
        channel_data = loader.load_channels(0, start_sample=0, end_sample=1000)
        print(f"   Shape: {channel_data.shape}")
        print(f"   First 10 values: {channel_data[:10, 0]}")
        print(f"   Mean: {channel_data.mean():.2f}, Std: {channel_data.std():.2f}")
        
        # Load multiple channels
        print("\n   Loading first 1000 samples from channels [0, 1, 2]...")
        multi_channel_data = loader.load_channels([0, 1, 2], start_sample=0, end_sample=1000)
        print(f"   Shape: {multi_channel_data.shape}")
        
    except Exception as e:
        print(f"   Error loading data: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\nUsage example:")
    print("  from eeg_loader import EEGLoader")
    print("  loader = EEGLoader('continuous.dat', num_channels=72)")
    print("  data = loader.load_channels([0, 1, 2], start_sample=0, end_sample=10000)")


if __name__ == "__main__":
    main()

