import SimpleITK as sitk
import matplotlib.pyplot as plt

# Load the 3D colorized volume
volume = sitk.ReadImage('colorized_3d_volume.nrrd')
volume_np = sitk.GetArrayFromImage(volume)  # Shape: (Slices, Height, Width, 3)

print(f"Volume shape: {volume_np.shape}")  # Should be (29, H, W, 3)

# View slices one by one
for idx in range(volume_np.shape[0]):
    plt.imshow(volume_np[idx])  # RGB image
    plt.title(f"Slice {idx}")
    plt.axis('off')
    plt.show()