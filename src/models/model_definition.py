import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

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