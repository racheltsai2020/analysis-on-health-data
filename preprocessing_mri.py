import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.registration import optical_flow_tvl1
from skimage.transform import resize
import nibabel as nib

mri_images = "cancer"

#creating mask for images
def create_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    binary_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=500)
    return binary_mask.astype(np.uint8)

#removing background
def remove_background(image, mask):
    return image*np.expand_dims(mask, axis=-1)

#reducing noise
def denoise_image(image):
    return cv2.fastNlMeansDenoising((image*255).astype(np.uint8), None, 10, 7,21) / 255.0

#resize image
def resample_image(image, target_shape):
    return resize(image, target_shape, order=3, mode= 'reflect', anti_aliasing=True)

def normalize_intensity(image, min_percentile=0.5, max_percentile=99.5):
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    return (image-min_val)/ (max_val - min_val)


for dataset_type in ["testing", "training"]:
    dataset_path = os.path.join(mri_images, dataset_type)

    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"Accessing folder: {subfolder_path}")

            image_files = [f for f in os.listdir(subfolder_path) if f.endswith((".png",".jpg",".jpeg"))]

            for img_file in image_files[:2]:
                img_path = os.path.join(subfolder_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                mask = create_mask(img)
                processed_img = remove_background(img, mask)
                improve_img = denoise_image(processed_img)
                #resizing makes some images weird so it is currently commented out
                #resize_img = resample_image(improve_img, (512,512))
                normalize_img = normalize_intensity(improve_img)

                # plt.imshow(normalize_img)
                # plt.title(f"Processed: {img_file}")
                # plt.axis("off")
                # plt.show()

fig, axes =plt.subplots(1, 2, figsize=(10,5))

#compare first images from the folders (original vs masked image)
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Binary Mask")
axes[1].axis("off")

plt.show()

#compare first images from the folders (original vs background removed image)
fig, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(img)
axes[0].set_title("Before Background removed")
axes[0].axis("off")

axes[1].imshow(processed_img)
axes[1].set_title("After Background Removal")
axes[1].axis("off")

fig, axes = plt.subplots(1, 2, figsize=(10,5))

#compare first images from the folders (original vs remove noise image)
axes[0].imshow(processed_img)
axes[0].set_title("Before Denoising")
axes[0].axis("off")

axes[1].imshow(improve_img)
axes[1].set_title("After Denoising")
axes[1].axis("off")
plt.show()

