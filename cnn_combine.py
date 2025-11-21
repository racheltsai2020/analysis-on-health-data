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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, LeakyReLU, ELU, SpatialDropout2D, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.efficientnet import EfficientNetB0
import time
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import datetime
import json

mri_images = "cancer"
train_folder = os.path.join(mri_images, "training")
test_folder = os.path.join(mri_images, "testing")
#mri_images = "Brain_Cancer"
#train_folder = mri_images
#test_folder = None
classes = sorted([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
num_class = len(classes)
os.makedirs("models", exist_ok=True)
with open("models/cnn_class.json", "w") as f:
    json.dump(classes, f)
print(f"Class order: {classes}")
#assert num_class ==4, f"found {num_class}: {classes}"
batch =32
AUTOTUNE = tf.data.AUTOTUNE

seed = 42
image_size = 256
np.random.seed(seed)
tf.random.set_seed(seed)

time_program = time.perf_counter()

tf.keras.backend.set_image_data_format('channels_last')


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
            raise ValueError(f"Unsupported channel count: {c}")
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

def load_and_preprocess(path, target_size, to_grayscale):
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
    inputs = tf.keras.layers.Input(shape=in_shape, name="cnn_input")
    
    x= Conv2D(32, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-5))(inputs)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = MaxPooling2D((2,2))(x)
    x = SpatialDropout2D(0.2)(x)

    x= Conv2D(64, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = MaxPooling2D((2,2))(x)
    x = SpatialDropout2D(0.25)(x)

    x= Conv2D(128, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = MaxPooling2D((2,2))(x)
    x = SpatialDropout2D(0.30)(x)

    x= Conv2D(256, (3,3), activation='relu', padding="same", kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = MaxPooling2D((2,2))(x)
    x = SpatialDropout2D(0.35)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax', name="cnn_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CustomCNN")
    model.compile(optimizer=Adam(1e-4), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model



train_img, train_label = [], []
for class_idx, class_name in enumerate(classes):
    class_path = os.path.join(train_folder, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith((".png", ".jpg",".jpeg")):
            fpath = os.path.join(class_path, fname)
            try:
                img = load_and_preprocess(fpath, target_size=(image_size,image_size,3), to_grayscale=False)
                if img is None or img.size ==0:
                    raise ValueError("Preprocess returned empty image")
                train_img.append(img)
                train_label.append(class_idx)
            except Exception as e:
                print(f"Warning: {e}")

train_img = np.array(train_img, dtype=np.float32)
train_label = np.array(train_label, dtype=np.int64)

class_counts = Counter(train_label)
print("Class counts:", class_counts)



plt.figure(figsize=(6,4))
plt.bar(class_counts.keys(), class_counts.values(), color='steelblue')
plt.xticks(list(class_counts.keys()), classes, rotation=15)
plt.title("Training class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

X_train, X_val, y_train, y_val = train_test_split(
        train_img, train_label,
        test_size=0.2,
        random_state=seed,
        stratify=train_label
    )

mean = np.mean(X_train)
std = np.std(X_train) + 1e-8

print(f"Global mean: {mean: .4f}, std: {std:.4f}")
print("training image shape:", X_train.shape)

#os.makedirs("models", exist_ok=True)
#np.save("models/mean_std.npy", [mean, std])

X_train = (X_train - mean)/ std
X_val = (X_val - mean) / std

X_test, y_test = [], []
test_classes = sorted([d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))])
for class_idx, class_name in enumerate(test_classes):
    class_path = os.path.join(test_folder, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith((".png", ".jpg",".jpeg")):
            fpath = os.path.join(class_path, fname)
            try:
                img = load_and_preprocess(fpath, target_size=(image_size, image_size, 3), to_grayscale=False)
                if img is None or img.size == 0:
                    raise ValueError("Preprocess returned empty image")
                X_test.append(img)
                y_test.append(class_idx)
            except Exception as e:
                print(f"warning (test set): {e}")

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int64)
X_test = (X_test - mean) / std

cw = compute_class_weight('balanced', classes= np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw))
print("class weight:", class_weights)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode ='nearest',
    preprocessing_function=lambda x: tf.image.random_contrast(x, 0.9, 1.1)
)

train_gen = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

#save best model
os.makedirs("models", exist_ok=True)
best_model = "models/cnn_combine_best.h5"

checkpoint = ModelCheckpoint(
    filepath = best_model,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1
)

#model = cnn_model((image_size,image_size,1))

#model.summary()
#history = model.fit(train_gen, epochs=30, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[lr, early_stop, checkpoint],verbose=1)

#val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
#test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
#print(f"Validation accuracy: {val_acc: .3f}")
#print(f"Test accuracy: {test_acc:.3f}")

#best_validation_accuracy = max(history.history["val_accuracy"])
#print(f"\n Best validation accuracy during training: {best_validation_accuracy:.4f}")

#with open("models/best_cnn_accuracy.txt", "w") as f:
    #f.write(str(best_validation_accuracy))


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    tf.keras.backend.set_image_data_format('channels_last')

    print("Keras image data format:", tf.keras.backend.image_data_format())

    import_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(image_size, image_size, 3), weights=None)
    weights_path = tf.keras.utils.get_file("efficientnetb0_notop.h5", "https://storage.googleapis.com/tensorflow/keras-applications/efficientnetb0/efficientnetb0_notop.h5", cache_subdir="models")
    import_model.load_weights(weights_path)
    import_model.trainable = False
    print("Input shape:", import_model.input_shape)

    cnn = cnn_model((image_size, image_size, 1))
    cnn.summary()
    print("CNN input:", cnn.input)
    print("CNN output", cnn.output)
    test = Model(inputs=cnn.input, outputs=cnn.layers[-3].output)
    feature_layer = cnn.layers[-3].name
    cnn = Model(inputs=cnn.input, outputs=cnn.get_layer(feature_layer).output)

    input_image = Input(shape=(image_size, image_size, 3), name="rgb_input")
    from tensorflow.keras.layers import Lambda
    input_gray_image = Lambda(lambda x:tf.image.rgb_to_grayscale(x), name="grayscale_layer")(input_image)

    pretrained_model = import_model(input_image, training=False)
    pretrained_model = GlobalAveragePooling2D()(pretrained_model)

    #process_cnn = cnn(input_gray_image)
    process_cnn = test(input_gray_image)
    #process_cnn = GlobalAveragePooling2D()(process_cnn)

    merged = Concatenate()([pretrained_model, process_cnn])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.4)(x)
    output = Dense(num_class, activation='softmax')(x)

    combination_model = Model(inputs=input_image, outputs=output)
    combination_model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    combination_model.summary()

    history = combination_model.fit(
        train_gen,
        epochs=30,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[lr, early_stop, checkpoint],
        verbose=1
    )

    val_loss, val_accuracy = combination_model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = combination_model.evaluate(X_test, y_test, verbose=0)

    print(f"Validation accuracy: {val_accuracy: .3f}")
    print(f"Test accuracy: {test_accuracy: .3f}")

    best_validation = max(history.history["val_accuracy"])
    print(f"\n best validation accuracy during training: {best_validation: .4f}")

    runtime = time.perf_counter() - time_program
    print(f"\n Total run time: {runtime: .2f} s")
