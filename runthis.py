


# Installing the necessary packages
# Suppress already satisfied warning using: | find /V "already satisfied"
# %pip install -r requirements.txt | find /V "already satisfied"

# Standard library imports
import os
from pathlib import Path
import warnings
import import_ipynb

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Custom utility functions
from utils import (
    load_images_and_masks,
    view_data,
    preprocess_images_and_masks,
    remap_mask_classes,
    save_datasets,
    load_datasets,
    save_associated_files,
    custom_warnings,
)

# importing these but not using unet_model anymore
from unet import (
    test_unet,
    plot_losses,
    save_predicted_masks,
    iou_metric,
    dice_coeff,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras import callbacks, optimizers
import tensorflow as tf

warnings.formatwarning = custom_warnings

PRODUCTION = False
seed_value = 42

if not PRODUCTION:
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

# directories
images_dir = "Datasets/Pala_Mesa_Roads_Rails/images"
masks_dir = "Datasets/Pala_Mesa_Roads_Rails/labels"
model_dir = "models"
model_name = "trained_UNet_Pala_Mesa.keras"
model_path = Path(f"{model_dir}/{model_name}")

# load data
images, masks, missing_masks, names_map = load_images_and_masks(
    images_dir,
    masks_dir,
    file_ext="tif",
    max_count=1000,
    trim_names=True
)

view_data(images=images, masks=masks, max_plots=10, max_cols=5, randomize=True, colors=True)

# preprocessing
num_classes = 4

preprocessed_images, preprocessed_masks, threshold, image_names, num_classes = (
    preprocess_images_and_masks(
        images,
        masks,
        num_classes=num_classes,
        threshold=0.5
    )
)

# add after preprocessing  

from scipy.ndimage import distance_transform_edt

# VEGETATION (NDVI)
# used chatgpt for this 
def compute_ndvi(images):
    ndvi_maps = []

    for img in images:
        red = img[:, :, 0].astype(np.float32)
        nir = img[:, :, 1].astype(np.float32)

        ndvi = (nir - red) / (nir + red + 1e-6)

        veg_class = np.zeros_like(ndvi)

        veg_class[ndvi < 0.2] = 0
        veg_class[(ndvi >= 0.2) & (ndvi < 0.5)] = 1
        veg_class[ndvi >= 0.5] = 2

        ndvi_maps.append(veg_class)

    return np.array(ndvi_maps)


# ROAD RISK
def compute_road_risk(masks):
    road_risk_maps = []

    for mask in masks:
        road_mask = (mask == 1).astype(np.uint8)

        distance = distance_transform_edt(1 - road_mask)
        distance = distance / np.max(distance)

        risk = np.zeros_like(distance)

        risk[distance < 0.1] = 3
        risk[(distance >= 0.1) & (distance < 0.3)] = 2
        risk[(distance >= 0.3) & (distance < 0.6)] = 1
        risk[distance >= 0.6] = 0

        road_risk_maps.append(risk)

    return np.array(road_risk_maps)


# generate labels
vegetation_maps = compute_ndvi(preprocessed_images)
road_risk_maps = compute_road_risk(preprocessed_masks)

# binary burn masks
binary_masks = (preprocessed_masks > 0).astype(np.uint8)

print("Vegetation maps:", vegetation_maps.shape)
print("Road risk maps:", road_risk_maps.shape)
print("Binary burn masks:", binary_masks.shape)


# split data
(
    images_train,
    images_test_and_val,
    burn_train,
    burn_test_and_val,
    veg_train,
    veg_test_and_val,
    road_train,
    road_test_and_val,
    names_train,
    names_test_and_val,
) = train_test_split(
    preprocessed_images,
    binary_masks,
    vegetation_maps,
    road_risk_maps,
    image_names,
    test_size=0.2,
    random_state=seed_value,
)

(
    images_validation,
    images_test,
    burn_validation,
    burn_test,
    veg_validation,
    veg_test,
    road_validation,
    road_test,
    names_validation,
    names_test,
) = train_test_split(
    images_test_and_val,
    burn_test_and_val,
    veg_test_and_val,
    road_test_and_val,
    names_test_and_val,
    test_size=0.5,
    random_state=seed_value,
)


# model 
from tensorflow.keras import layers, Model

def multi_output_unet(input_shape=(256,256,3)):

    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)

    burn_output = layers.Conv2D(1, 1, activation="sigmoid", name="burn_output")(c4)
    veg_output = layers.Conv2D(3, 1, activation="softmax", name="veg_output")(c4)
    road_output = layers.Conv2D(4, 1, activation="softmax", name="road_output")(c4)

    return Model(inputs=inputs, outputs=[burn_output, veg_output, road_output])


model = multi_output_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "burn_output": "binary_crossentropy",
        "veg_output": "sparse_categorical_crossentropy",
        "road_output": "sparse_categorical_crossentropy",
    },
    metrics={
        "burn_output": ["accuracy"],
        "veg_output": ["accuracy"],
        "road_output": ["accuracy"],
    }
)

model.summary()


# training
model_fit = model.fit(
    images_train,
    {
        "burn_output": burn_train,
        "veg_output": veg_train,
        "road_output": road_train,
    },
    batch_size=8,
    epochs=25,
    validation_data=(
        images_validation,
        {
            "burn_output": burn_validation,
            "veg_output": veg_validation,
            "road_output": road_validation,
        }
    ),
    verbose=1
)


# predictions
preds = model.predict(images_test)

burn_preds = preds[0]
veg_preds = preds[1]
road_preds = preds[2]


# visualization
def show_results(idx):
    plt.figure(figsize=(15,5))

    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(images_test[idx][:,:,:3])

    plt.subplot(1,4,2)
    plt.title("Burn Prediction")
    plt.imshow((burn_preds[idx] > 0.5).astype(int))

    plt.subplot(1,4,3)
    plt.title("Vegetation")
    plt.imshow(np.argmax(veg_preds[idx], axis=-1))

    plt.subplot(1,4,4)
    plt.title("Road Risk")
    plt.imshow(np.argmax(road_preds[idx], axis=-1))

    plt.show()

show_results(0)