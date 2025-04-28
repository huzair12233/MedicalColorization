import torch
import urllib
import cv2
import numpy as np
import open3d as o3d
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Download pretrained MiDaS model
model_type = "MiDaS_small" # lightweight
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load your colorized medical image
img = cv2.imread("output/colorized/colorized_1.jpeg")  # your path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Prepare image for MiDaS
input_batch = transform(img_rgb).unsqueeze(0)

# Predict depth
with torch.no_grad():
    prediction = midas(input_batch)

depth_map = prediction.squeeze().cpu().numpy()

# Normalize depth for better visualization
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]))

# Create point cloud
h, w = depth_map.shape
i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
x = i.flatten()
y = j.flatten()
z = depth_map.flatten() * 100  # scale depth if needed

points = np.vstack((x, y, z)).T
colors = img_rgb.reshape(-1, 3) / 255.0

# Create Open3D PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save point cloud
o3d.io.write_point_cloud("output/3dmodels/colorized_depth_3d.ply", pcd)

# Visualize
o3d.visualization.draw_geometries([pcd])