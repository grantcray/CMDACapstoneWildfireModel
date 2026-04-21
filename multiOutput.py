



# add after preprocessing 

# import for distance calc
from scipy.ndimage import distance_transform_edt
# calculates how far from roads



# VEGETATION (NDVI)
# used chatgpt for this 
def compute_ndvi(images):
    ndvi_maps = []  # stores vegetation maps for all images

    for img in images:
        # Grab two color channels from the image
        red = img[:, :, 0].astype(np.float32)  # Red light
        nir = img[:, :, 1].astype(np.float32)  # Near Infrared (plants reflect this)

        # NDVI formula ...  how much vegetation is present
        ndvi = (nir - red) / (nir + red + 1e-6)

        # Create empty map (same size as image)
        veg_class = np.zeros_like(ndvi)

        # Turn NDVI into categories (simple classification)
        veg_class[ndvi < 0.2] = 0        # little/no vegetation
        veg_class[(ndvi >= 0.2) & (ndvi < 0.5)] = 1  # medium vegetation
        veg_class[ndvi >= 0.5] = 2       # dense vegetation

        ndvi_maps.append(veg_class)

    return np.array(ndvi_maps)  # return all vegetation maps





# ROAD RISK
def compute_road_risk(masks):
    road_risk_maps = []

    for mask in masks:
        # Find where roads are (assuming roads = class 1) 
        # figure this out... what are roads classified as  
        road_mask = (mask == 1).astype(np.uint8)

        # Calculate distance to nearest road pixel
        distance = distance_transform_edt(1 - road_mask)

        # Normalize (scale between 0 and 1)
        distance = distance / np.max(distance)

        # Create empty risk map
        risk = np.zeros_like(distance)

        # Assign risk levels based on distance
        # might need to adjust these
        risk[distance < 0.1] = 3   # very close = high risk
        risk[(distance >= 0.1) & (distance < 0.3)] = 2
        risk[(distance >= 0.3) & (distance < 0.6)] = 1
        risk[distance >= 0.6] = 0  # far away = low risk

        road_risk_maps.append(risk)

    return np.array(road_risk_maps)


# GENERATE NEW LABELS
# Create vegetation labels from images
vegetation_maps = compute_ndvi(preprocessed_images)
# Create road risk labels from masks
road_risk_maps = compute_road_risk(preprocessed_masks)

# burn masks to binary (fire vs no fire)
# anything > 0 becomes fire (1)
binary_masks = (preprocessed_masks > 0).astype(np.uint8)

print("Vegetation maps:", vegetation_maps.shape)
print("Road risk maps:", road_risk_maps.shape)
print("Binary burn masks:", binary_masks.shape)


# SPLIT DATA

# Split into training + temp set
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

# Split temp into validation + test
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





# MODEL

from tensorflow.keras import layers, Model

def multi_output_unet(input_shape=(256,256,3), num_classes=4):

    inputs = layers.Input(shape=input_shape)

    # encoder
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

    # binary burn output
    burn_output = layers.Conv2D(
        1, 1, activation="sigmoid", name="burn_output"
    )(c4)

    # vegetation output 
    veg_output = layers.Conv2D(
        3, 1, activation="softmax", name="veg_output"
    )(c4)

    # road risk output 
    road_output = layers.Conv2D(
        4, 1, activation="softmax", name="road_output"
    )(c4)

    return Model(inputs=inputs, outputs=[burn_output, veg_output, road_output])



# compile model

model = multi_output_unet(input_shape=(256,256,3), num_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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

burn_preds = preds[0]  # fire predictions
veg_preds = preds[1]  # vegetation predictions
road_preds = preds[2]  # road risk predictions


# visuals 

def show_results(idx):
    plt.figure(figsize=(15,5))

    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(images_test[idx][:,:,:3])

    plt.subplot(1,4,2)
    plt.title("Burn Prediction")
    plt.imshow((burn_preds[idx] > 0.5).astype(int))  #threshold instead of argmax... maybe edit

    plt.subplot(1,4,3)
    plt.title("Vegetation")
    plt.imshow(np.argmax(veg_preds[idx], axis=-1))

    plt.subplot(1,4,4)
    plt.title("Road Risk")
    plt.imshow(np.argmax(road_preds[idx], axis=-1))

    plt.show()

show_results(0)