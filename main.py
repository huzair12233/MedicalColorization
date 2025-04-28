import os
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ====== CONFIG ====== The file path changes as per the dataset
INPUT_NRRD_PATH = "input/R-002-1.nrrd"  # <-- Change this to your grayscale .nrrd file
OUTPUT_NRRD_PATH = "output/colorized_3d_volume.nrrd"
TMP_SLICE_DIR = "temp_slices/"
TMP_COLORIZED_DIR = "temp_colorized_slices/"

# Model file
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
cluster_path = "models/pts_in_hull.npy"

if not os.path.exists(OUTPUT_NRRD_PATH):
    os.makedirs("output", exist_ok=True)

# ====== SETUP ======
os.makedirs(TMP_SLICE_DIR, exist_ok=True)
os.makedirs(TMP_COLORIZED_DIR, exist_ok=True)

# Load colorization model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(cluster_path)
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# ====== STEP 1: Load and Extract slices ======
print("🔵 Loading NRRD volume...")
volume = sitk.ReadImage(INPUT_NRRD_PATH)
volume_np = sitk.GetArrayFromImage(volume)  # (Depth, Height, Width)

print(f"Volume shape: {volume_np.shape}")
print("🔵 Extracting slices...")

for idx, slice_ in enumerate(volume_np):
    slice_path = os.path.join(TMP_SLICE_DIR, f"slice_{idx:03d}.png")
    plt.imsave(slice_path, slice_, cmap='gray')

print("✅ Slices saved.")

# ====== STEP 2: Colorize each slice ======
print("🟠 Colorizing slices...")

for image_file in sorted(os.listdir(TMP_SLICE_DIR)):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(TMP_SLICE_DIR, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load {image_file}")
            continue

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]

        resized_L = cv2.resize(L, (224, 224))
        resized_L -= 50

        net.setInput(cv2.dnn.blobFromImage(resized_L))
        ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab_dec_us = cv2.resize(ab_dec, (image.shape[1], image.shape[0]))
        L = L[:, :, np.newaxis]
        Lab_out = np.concatenate((L, ab_dec_us), axis=2)

        bgr_out = cv2.cvtColor(Lab_out.astype("float32"), cv2.COLOR_LAB2BGR)
        bgr_out = np.clip(bgr_out, 0, 1)

        output_path = os.path.join(TMP_COLORIZED_DIR, f"colorized_{image_file}")
        cv2.imwrite(output_path, (bgr_out * 255).astype("uint8"))

print("✅ All slices colorized.")

# ====== STEP 3: Stack colorized slices into volume ======
print("🟢 Rebuilding colorized 3D volume...")

colorized_slices = []
for filename in sorted(os.listdir(TMP_COLORIZED_DIR)):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(TMP_COLORIZED_DIR, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colorized_slices.append(img)

volume_color_np = np.stack(colorized_slices, axis=0)  # Shape: (Depth, Height, Width, 3)

# Save as NRRD
volume_color_sitk = sitk.GetImageFromArray(volume_color_np, isVector=True)
sitk.WriteImage(volume_color_sitk, OUTPUT_NRRD_PATH)

print(f"✅ Final colorized 3D volume saved at: {OUTPUT_NRRD_PATH}")

# ====== CLEANUP (optional) ======
import shutil
shutil.rmtree(TMP_SLICE_DIR)
shutil.rmtree(TMP_COLORIZED_DIR)