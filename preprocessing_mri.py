import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.registration import optical_flow_tvl1
from skimage.transform import resize
import nibabel as nib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import time

from tensorflow.python.ops.gen_nn_ops import LeakyRelu

mri_images = "cancer"
train_folder = os.path.join(mri_images, "training")
test_folder = os.path.join(mri_images, "testing")
classes = sorted([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
num_class = len(classes)
assert num_class ==4, f"found {num_class}: {classes}"
batch =32
AUTOTUNE = tf.data.AUTOTUNE

seed = 42
image_size = 512
np.random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

time_program = time.perf_counter()


def to_grayscale(image):
    if image is None:
        raise ValueError("Image is None (bad path or read error).")

    if image.ndim == 2:
        gray = image
    elif image.ndim ==3:
        c = image.shape[2]
        if c == 1:
            gray = image[:,:, 0]
        elif c==3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif c==4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupport channel count: {c}")
    else:
        raise ValueError(f"unsupported image shapre: {image.shape}")

    if gray.dtype != np.uint8:
        gmin, gmax = float(np.min(gray)), float(np.max(gray))
        gray = ((gray - gmin) / (gmax-gmin)*255.0).astype(np.uint8) if gmax > gmin else np.zeros_like(gray, dtype=np.uint8)

    return gray


#creating mask for images
def create_mask(image):
    gray = to_grayscale(image)
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.medianBlur(binary_mask, 3)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return binary_mask

#removing background
def remove_background(image, mask):
    m= (mask > 0).astype(image.dtype)
    if image.ndim ==2:
        return image*m
    elif image.ndim == 3:
        return image *m[..., None]
    else: 
        raise ValueError(f"image shape unsupported: {image.shape}")
    return image*np.expand_dims(mask, axis=-1)

#reducing noise
def denoise_image(image):
    x= image
    #img = (np.clip(image, 0,1)*255).astype(np.uint8)
    if x.ndim == 3 and x.shape[-1]==1:
        x = x[...,0]
    img = (np.clip(x,0,1)*255).astype(np.uint8)
    out = cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return (out.astype(np.float32) / 255.0)[..., None]

#resize image
def resample_image(image, target_shape):
    return resize(image, target_shape, order=3, mode= 'reflect', anti_aliasing=True)

def normalize_intensity(image, min_percentile=0.5, max_percentile=99.5):
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    eps = 1e-6
    return (image-min_val)/ (max_val - min_val+eps)

def resize_img(image, target_size):
    return resize(image, (target_size[0],target_size[1]), order=3, mode='reflect', anti_aliasing=True)

def load_and_preprocess(path, target_size, to_grayscale=True):
    if isinstance(target_size, (list, tuple)) and len(target_size) ==3:
        target_size = (target_size[0], target_size[1])
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    #mask
    mask = create_mask(rgb)
    rgb_bg_remove = remove_background(rgb, mask)

    gray = cv2.cvtColor((rgb_bg_remove*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/ 255.0
    gray = gray[..., None]

    #Denoise & normalize
    #rgb_denoise = denoise_image(rgb_bg_remove.astype(np.float32)/255.0)
    rgb_norm = normalize_intensity(gray)

    #resize
    rgb_resize = resize(rgb_norm, (target_size[0], target_size[1], 1),
                        order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
    if to_grayscale:
        return rgb_resize
    else:
        return np.repeat(rgb_resize, 3, axis=-1)

def cnn_model(in_shape):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-4), input_shape= in_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    #model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
model = cnn_model((image_size,image_size,1))

model.summary()

train_img, train_label = [], []
for class_idx, class_name in enumerate(classes):
    class_path = os.path.join(train_folder, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith((".png", ".jpg",".jpeg")):
            fpath = os.path.join(class_path, fname)
            try:
                img = load_and_preprocess(fpath, target_size=(image_size,image_size,1), to_grayscale=True)
                if img is None or img.size ==0:
                    raise ValueError("Preprocess returned empty image")
                train_img.append(img)
                train_label.append(class_idx)
            except Exception as e:
                print(f"Warning: {e}")

train_img = np.array(train_img, dtype=np.float32)
train_label = np.array(train_label, dtype=np.int64)

X_train, X_val, y_train, y_val = train_test_split(
        train_img, train_label,
        test_size=0.2,
        random_state=seed,
        stratify=train_label
    )

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode ='nearest')

train_gen = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
history = model.fit(train_gen, epochs=30, validation_data=(X_val, y_val),callbacks=[lr, early_stop],verbose=1)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc: .3f}")


for dataset_type in ["testing", "training"]:
    dataset_path = os.path.join(mri_images, dataset_type)

    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"Accessing folder: {subfolder_path}")

            image_files = [f for f in os.listdir(subfolder_path) if f.endswith((".png",".jpg",".jpeg"))]

            for img_file in image_files[:2]:
                img_path = os.path.join(subfolder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                mask = create_mask(img)
                processed_img = remove_background(img, mask)
                improve_img = denoise_image(processed_img)
                #resizing makes some images weird so it is currently commented out
                resize_img = resample_image(improve_img, (image_size,image_size, 1))
                normalize_img = normalize_intensity(resize_img)

elapsed_all = time.perf_counter() - time_program
print(f"\n Total run time: {elapsed_all: .2f} s")

                


                # plt.imshow(normalize_img)
                # plt.title(f"Processed: {img_file}")
                # plt.axis("off")
                # plt.show()

#fig, axes =plt.subplots(1, 2, figsize=(10,5))

#compare first images from the folders (original vs masked image)
#axes[0].imshow(img)
#axes[0].set_title("Original Image")
#axes[0].axis("off")

#axes[1].imshow(mask, cmap="gray")
#axes[1].set_title("Binary Mask")
#axes[1].axis("off")

#plt.show()

#compare first images from the folders (original vs background removed image)
#fig, axes = plt.subplots(1,2, figsize=(10,5))
#axes[0].imshow(img)
#axes[0].set_title("Before Background removed")
#axes[0].axis("off")

#axes[1].imshow(processed_img)
#axes[1].set_title("After Background Removal")
#axes[1].axis("off")

#fig, axes = plt.subplots(1, 2, figsize=(10,5))

#compare first images from the folders (original vs remove noise image)
#axes[0].imshow(processed_img)
#axes[0].set_title("Before Denoising")
#axes[0].axis("off")

#axes[1].imshow(improve_img)
#axes[1].set_title("After Denoising")
#axes[1].axis("off")
#plt.show()


