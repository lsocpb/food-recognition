import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.applications.vgg16 import preprocess_input

# GLOBAL VARIABLES IN THE SCOPE OF THE FILE
IMG_SIZE = 224
TARGET_CLASSES = [
    'baby_back_ribs',
    'breakfast_burrito',
    'hamburger',
    'pancakes',
    'pizza',
    'risotto',
    'steak',
    'spaghetti_bolognese'
]
NUM_CLASSES = len(TARGET_CLASSES)
# MAP CLASS NAMES INTO TENSORFLOW INDEXES
CLASS_MAP = {name: i for i, name in enumerate(TARGET_CLASSES)}

def filter_classes(image, label, all_class_names):
    def _filter_and_remap(image, label):
        label_int = label.numpy().astype(np.int64)
        label_name = all_class_names[label_int] 
        
        if label_name in TARGET_CLASSES:
            new_label_val = CLASS_MAP[label_name]
            return np.array(True), image, np.array(new_label_val, dtype=np.int64) 
        else:
            return np.array(False), image, label
            
    is_in_target, image, new_label = tf.py_function(
        _filter_and_remap, 
        inp=[image, label], 
        Tout=[tf.bool, tf.uint8, tf.int64]
    )
    
    is_in_target.set_shape([])
    new_label.set_shape([])
    image.set_shape([None, None, 3])
    
    return is_in_target, image, new_label

def process_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label
    
def process_image_for_vgg(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

def apply_filter_map(is_in_target, image, new_label):
    return image, new_label

def load_and_split_data(batch_size=32):
    """
    THIS LOADS FOOD-101, FILTERS TO 8 CLASSES DEFINED IN TARGET_CLASSES AND DIVIDES INTO CITES
    """
    (ds_train, ds_validation, ds_test), ds_info = tfds.load(
        'food101',
        split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
        as_supervised=True,
        with_info=True
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2)
    ])

    def apply_augmentation(image, label):
        return data_augmentation(image, training=True), label
    
    all_class_names = ds_info.features['label'].names
    
    ds_train_filtered = ds_train.map(lambda x, y: filter_classes(x, y, all_class_names), num_parallel_calls=tf.data.AUTOTUNE)
    ds_val_filtered = ds_validation.map(lambda x, y: filter_classes(x, y, all_class_names), num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_filtered = ds_test.map(lambda x, y: filter_classes(x, y, all_class_names), num_parallel_calls=tf.data.AUTOTUNE)

    ds_train_filtered = ds_train_filtered.filter(lambda is_in_target, image, new_label: is_in_target)
    ds_val_filtered = ds_val_filtered.filter(lambda is_in_target, image, new_label: is_in_target)
    ds_test_filtered = ds_test_filtered.filter(lambda is_in_target, image, new_label: is_in_target)

    ds_train_final = ds_train_filtered.map(apply_filter_map)
    ds_val_final = ds_val_filtered.map(apply_filter_map)
    ds_test_final = ds_test_filtered.map(apply_filter_map)
    
    train_ds = ds_train_final.map(process_image).map(apply_augmentation).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = ds_val_final.map(process_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = ds_test_final.map(process_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, ds_info

def load_and_split_data_for_vgg16(batch_size=32):
    (ds_train, ds_validation, ds_test), ds_info = tfds.load(
        'food101',
        split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
        as_supervised=True,
        with_info=True
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2)
    ])

    def apply_augmentation(image, label):
        return data_augmentation(image, training=True), label
    
    all_class_names = ds_info.features['label'].names
    
    ds_train_filtered = ds_train.map(lambda x, y: filter_classes(x, y, all_class_names), num_parallel_calls=tf.data.AUTOTUNE)
    ds_val_filtered = ds_validation.map(lambda x, y: filter_classes(x, y, all_class_names), num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_filtered = ds_test.map(lambda x, y: filter_classes(x, y, all_class_names), num_parallel_calls=tf.data.AUTOTUNE)

    ds_train_filtered = ds_train_filtered.filter(lambda is_in_target, image, new_label: is_in_target)
    ds_val_filtered = ds_val_filtered.filter(lambda is_in_target, image, new_label: is_in_target)
    ds_test_filtered = ds_test_filtered.filter(lambda is_in_target, image, new_label: is_in_target)

    ds_train_final = ds_train_filtered.map(apply_filter_map)
    ds_val_final = ds_val_filtered.map(apply_filter_map)
    ds_test_final = ds_test_filtered.map(apply_filter_map)
    
    train_ds = ds_train_final.map(process_image_for_vgg).map(apply_augmentation).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = ds_val_final.map(process_image_for_vgg).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = ds_test_final.map(process_image_for_vgg).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, ds_info