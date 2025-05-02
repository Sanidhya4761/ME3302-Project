import numpy as np
import cv2
import os
import csv
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
image_size = 128
num_images_per_class = 500
output_dir = "thermal_defect_dataset_rgb"
defect_dir = os.path.join(output_dir, "images", "defect")
no_defect_dir = os.path.join(output_dir, "images", "no_defect")
os.makedirs(defect_dir, exist_ok=True)
os.makedirs(no_defect_dir, exist_ok=True)

def generate_background():
    """Generate a smooth thermal background with noise and gradient."""
    base_temp = np.random.uniform(0.4, 0.6)
    gradient = np.tile(np.linspace(base_temp - 0.1, base_temp + 0.1, image_size), (image_size, 1))
    noise = np.random.normal(0, 0.02, (image_size, image_size))
    thermal_gray = np.clip(gradient + noise, 0, 1)
    return thermal_gray

def add_defect(image):
    """Add an elliptical defect with temperature deviation."""
    defect_img = image.copy()
    center = tuple(np.random.randint(20, image_size - 20, size=2))
    axes = tuple(np.random.randint(5, 15, size=2))
    angle = np.random.randint(0, 180)
    intensity = np.random.uniform(0.2, 0.4) * (1 if np.random.rand() > 0.5 else -1)
    color = float(np.clip(defect_img[center[1], center[0]] + intensity, 0, 1))
    cv2.ellipse(defect_img, center, axes, angle, 0, 360, color, -1)
    return defect_img

def to_colormap(img_gray):
    """Convert normalized gray image to RGB using colormap."""
    img_uint8 = (img_gray * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
    return img_color

# Create label CSV
label_file = os.path.join(output_dir, "labels.csv")
with open(label_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    # Generate defect images
    for i in tqdm(range(num_images_per_class), desc="Generating defect images"):
        gray = generate_background()
        defect_img = add_defect(gray)
        rgb_img = to_colormap(defect_img)
        filename = f"img_{i:04d}.jpg"
        cv2.imwrite(os.path.join(defect_dir, filename), rgb_img)
        writer.writerow([f"images/defect/{filename}", 1])

    # Generate no-defect images
    for i in tqdm(range(num_images_per_class), desc="Generating no-defect images"):
        gray = generate_background()
        rgb_img = to_colormap(gray)
        filename = f"img_{i+1000:04d}.jpg"
        cv2.imwrite(os.path.join(no_defect_dir, filename), rgb_img)
        writer.writerow([f"images/no_defect/{filename}", 0])

print(f"\n✅ RGB thermal dataset saved to: {output_dir}")
