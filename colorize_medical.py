import cv2
import numpy as np
import os

# Paths to model files
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
cluster_path = "models/pts_in_hull.npy"

# Input and output directories
input_dir = "input/"
output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the pretrained model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load cluster centers
pts = np.load(cluster_path)
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Process each image
for image_file in os.listdir(input_dir):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        # Read the grayscale image
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_file}")
            continue

        # Convert to Lab color space
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]  # L channel

        # Resize to 224x224 as model expects
        resized_L = cv2.resize(L, (224, 224))
        resized_L -= 50  # Mean-centering as per model requirement

        net.setInput(cv2.dnn.blobFromImage(resized_L))
        ab_dec = net.forward()[0, :, :, :].transpose((1,2,0))  # ab channels

        # Resize ab channels to original image size
        ab_dec_us = cv2.resize(ab_dec, (image.shape[1], image.shape[0]))

        # Concatenate L and ab
        L = L[:, :, np.newaxis]
        Lab_out = np.concatenate((L, ab_dec_us), axis=2)

        # Convert to BGR
        bgr_out = cv2.cvtColor(Lab_out.astype("float32"), cv2.COLOR_LAB2BGR)
        bgr_out = np.clip(bgr_out, 0, 1)

        # Save output
        output_path = os.path.join(output_dir, f"colorized_{image_file}")
        cv2.imwrite(output_path, (bgr_out*255).astype("uint8"))
        print(f"✅ Saved colorized image to {output_path}")