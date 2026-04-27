# Installing the necessary packages
# (only run this once if needed)
# %pip install -r requirements.txt

# basic imports
import os
from pathlib import Path
import warnings
import import_ipynb

# plotting + math stuff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# helper functions from utils notebook
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

# still importing these even though we aren't using unet_model anymore
from unet import (
    test_unet,
    plot_losses,
    save_predicted_masks,
    iou_metric,
    dice_coeff,
)

# hide tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras import callbacks, optimizers
import tensorflow as tf

warnings.formatwarning = custom_warnings

# setting seeds so results are repeatable
PRODUCTION = False
seed_value = 42

if not PRODUCTION:
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

# file paths
images_dir = "Datasets/Pala_Mesa_Roads_Rails/images"
masks_dir = "Datasets/Pala_Mesa_Roads_Rails/labels"
model_dir = "models"
model_name = "trained_UNet_Pala_Mesa.keras"
model_path = Path(f"{model_dir}/{model_name}")

# load images + masks
images, masks, missing_masks, names_map = load_images_and_masks(
    images_dir,
    masks_dir,
    file_ext="tif",
    max_count=1000,
    trim_names=True
)

# examples testing
view_data(images=images, masks=masks, max_plots=10, max_cols=5, randomize=True, colors=True)

# preprocessing step 
num_classes = 4

preprocessed_images, preprocessed_masks, threshold, image_names, num_classes = (
    preprocess_images_and_masks(
        images,
        masks,
        num_classes=num_classes,
        threshold=0.5
    )
)




from scipy.ndimage import distance_transform_edt  # used to compute distance from roads

# VEGETATION (NDVI)
# how green something is
def compute_ndvi(images):
    ndvi_maps = []

    for img in images:
        # grabbing red + near infrared bands
        red = img[:, :, 0].astype(np.float32)
        nir = img[:, :, 1].astype(np.float32)

        # NDVI formula
        ndvi = (nir - red) / (nir + red + 1e-6)

        # turning NDVI into categories
        veg_class = np.zeros_like(ndvi)

        veg_class[ndvi < 0.2] = 0  # low vegetation
        veg_class[(ndvi >= 0.2) & (ndvi < 0.5)] = 1  # medium
        veg_class[ndvi >= 0.5] = 2 # dense

        ndvi_maps.append(veg_class)

    return np.array(ndvi_maps)


# road risk
# closer to roads = higher risk
def compute_road_risk(masks):
    road_risk_maps = []

    for mask in masks:
        # assume roads are labeled as class 1 --> fix this
        road_mask = (mask == 1).astype(np.uint8)

        # distance from road
        distance = distance_transform_edt(1 - road_mask)
        distance = distance / np.max(distance)

        # convert distance to risk level
        risk = np.zeros_like(distance)

        risk[distance < 0.1] = 3   # very close = high risk
        risk[(distance >= 0.1) & (distance < 0.3)] = 2
        risk[(distance >= 0.3) & (distance < 0.6)] = 1
        risk[distance >= 0.6] = 0  # far away = low risk

        road_risk_maps.append(risk)

    return np.array(road_risk_maps)


# generate extra labels
vegetation_maps = compute_ndvi(preprocessed_images)
road_risk_maps = compute_road_risk(preprocessed_masks)

# convert fire to binary (fire vs no fire)
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


# new model attempt

from tensorflow.keras import layers, Model

def multi_output_unet(input_shape=(256,256,3)):

    inputs = layers.Input(shape=input_shape)

    # encoder - extract features
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    # bottleneck
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    # decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)

    # outputs
    burn_output = layers.Conv2D(1, 1, activation="sigmoid", name="burn_output")(c4)
    veg_output = layers.Conv2D(3, 1, activation="softmax", name="veg_output")(c4)
    road_output = layers.Conv2D(4, 1, activation="softmax", name="road_output")(c4)

    return Model(inputs=inputs, outputs=[burn_output, veg_output, road_output])


model = multi_output_unet()

# compiling model
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


# train
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


# predict
preds = model.predict(images_test)

burn_preds = preds[0]
veg_preds = preds[1]
road_preds = preds[2]


# visuals 
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
