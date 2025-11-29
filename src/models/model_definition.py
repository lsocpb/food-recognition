import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, VGG16, vgg16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization

from src.preprocessing.preprocessing import NUM_CLASSES

def create_transfer_model(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet" 
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), 
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax")
    ], name="EfficientNet_Transfer_Model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model
def create_vgg16_transfer_model(input_shape=(224,224,3)):
    base_model = VGG16(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ], name="VGG16_Transfer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def unfreeze_last_n_layers(model, fine_tune_at=2, lr=1e-5):
    base_model = model.layers[0]
    base_model.trainable = True

    n_layers = len(base_model.layers)
    if fine_tune_at <= 0:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        fine_tune_at = min(fine_tune_at, n_layers)
        for layer in base_model.layers[: -fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_at:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

