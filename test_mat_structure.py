"""
Test script to inspect MAT file structure
"""
import os
import sys

# Try to find the MAT file
mat_file = None
search_paths = [
    '/home/user/ai',
    '/home/user',
    os.path.expanduser('~')
]

for search_path in search_paths:
    for root, dirs, files in os.walk(search_path):
        for f in files:
            if '4AP' in f and 'Illumination' in f and f.endswith('.mat'):
                mat_file = os.path.join(root, f)
                print(f'Found: {mat_file}')
                break
        if mat_file:
            break
    if mat_file:
        break

if not mat_file:
    print("MAT file not found!")
    print("\nSearching for any .mat files...")
    for search_path in ['/home/user/ai', '/home/user']:
        for root, dirs, files in os.walk(search_path):
            for f in files:
                if f.endswith('.mat'):
                    print(f"  {os.path.join(root, f)}")
    sys.exit(1)

# Try different methods to load
print("\n" + "="*60)
print("METHOD 1: Using h5py (MATLAB v7.3+)")
print("="*60)
try:
    import h5py
    with h5py.File(mat_file, 'r') as f:
        print("Keys:", list(f.keys()))
        for key in f.keys():
            print(f"\n{key}:")
            print(f"  Type: {type(f[key])}")
            if hasattr(f[key], 'shape'):
                print(f"  Shape: {f[key].shape}")
            if hasattr(f[key], 'dtype'):
                print(f"  Dtype: {f[key].dtype}")
except Exception as e:
    print(f"h5py failed: {e}")

print("\n" + "="*60)
print("METHOD 2: Using mat73")
print("="*60)
try:
    import mat73
    data = mat73.loadmat(mat_file)
    print("Keys:", list(data.keys()))
    for key in data.keys():
        if not key.startswith('__'):
            print(f"\n{key}:")
            print(f"  Type: {type(data[key])}")
            if hasattr(data[key], 'shape'):
                print(f"  Shape: {data[key].shape}")
            if hasattr(data[key], 'dtype'):
                print(f"  Dtype: {data[key].dtype}")
except Exception as e:
    print(f"mat73 not available or failed: {e}")

print("\n" + "="*60)
print("FILE INFO")
print("="*60)
print(f"File size: {os.path.getsize(mat_file) / 1024 / 1024:.2f} MB")
