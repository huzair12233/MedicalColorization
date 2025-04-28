# extract_slices.py
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

input_volume_path = "16396/Case9_US_resampled-label.nrrd"
slices_output_dir = "input/"

# Make sure input/ directory exists
os.makedirs(slices_output_dir, exist_ok=True)

# Load the 3D volume
volume = sitk.ReadImage(input_volume_path)
volume_np = sitk.GetArrayFromImage(volume)  # (Slices, Height, Width)

print(f"Loaded volume shape: {volume_np.shape}")

# Save each slice as a grayscale .png
for idx, slice in enumerate(volume_np):
    plt.imsave(f"{slices_output_dir}/slice_{idx:03d}.png", slice, cmap='gray')

print("✅ All slices saved in 'input/' folder.")