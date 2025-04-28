import open3d as o3d
import numpy as np
import cv2
import os

# Load the colorized image
colorized_image_path = "output/colorized/colorized_2.png"  # Replace with your actual path
img = cv2.imread(colorized_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width = gray.shape

# Create meshgrid for X and Y
X, Y = np.meshgrid(np.arange(width), np.arange(height))

# Normalize depth (Z-axis) from grayscale brightness
Z = gray.astype(np.float32) / 255.0 * 50  # Scale Z height (you can adjust 50)

# Flatten everything for Open3D
points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

# Get colors from original colorized image
colors = img_rgb.reshape(-1, 3) / 255.0  # Normalize RGB colors to 0-1

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Optional: create a mesh from the point cloud
# This creates a simple "ball-pivoting" surface if you want surface instead of points
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=5.0)

# Save the point cloud (or mesh)
output_dir = "output/3dmodels/"
os.makedirs(output_dir, exist_ok=True)

o3d.io.write_point_cloud(os.path.join(output_dir, "colorized_3d_model.ply"), pcd)
# If you create mesh, you can also save:
# o3d.io.write_triangle_mesh(os.path.join(output_dir, "colorized_3d_model.obj"), mesh)

# Visualize
o3d.visualization.draw_geometries([pcd])