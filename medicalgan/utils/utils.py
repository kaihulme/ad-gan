import os
import math
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


def normalise_dataset(x, y):
    return x / 255.0, y


def get_dataset(data_dir, rows, cols, batch_size, label_mode=None, shuffle=True, color_mode="grayscale"):
    """
    Creates tf.data object from directory.
    """
    target_dir = os.path.join(os.getcwd(), data_dir)
    data = image_dataset_from_directory(
        target_dir, 
        label_mode=label_mode, 
        image_size=(rows, cols), 
        color_mode=color_mode,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    if label_mode:
        return data.map(normalise_dataset)
    return data.map(lambda x: x / 255.0)


def get_aug_dataset(data_dir, rows, cols, batch_size, label_mode=None, shuffle=True, color_mode="grayscale"):
    target_dir = os.path.join(os.getcwd(), data_dir)
    aug_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        horizontal_flip=True,
        # other augmentations not viable due to mri normalisation
    )
    aug_data = aug_datagen.flow_from_directory(
        target_dir,
        class_mode=label_mode, 
        target_size=(rows, cols), 
        color_mode=color_mode,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return aug_data


def get_data_length(data_dir):
    """
    Gets the number of samples in dataset.
    """
    class_dirs = [os.listdir(os.path.join(data_dir, class_dir)) for class_dir in os.listdir(data_dir)]
    data_length = 0
    for class_dir in class_dirs:
        data_length += len(class_dir)
    return data_length


def get_model(dataset, model_type, model_note):
    model_dir = "out/models/{0}/{1}/{2}".format(dataset, model_type, model_note)
    latest_model = sorted(os.listdir(model_dir))[-1]
    model_path = os.path.join(model_dir, latest_model)
    return load_model(model_path)


def get_model_path(dataset, model_type, model_note):
    """
    Gets path for saving model.
    """
    model_name = time.strftime("model_%Y-%m-%d_%H:%M:%S")
    model_dir = "out/models/{0}/{1}/{2}".format(dataset, model_type, model_note)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    return os.path.join(model_dir, model_name)


def get_tb_dir(dataset, model_type, model_note):
    """
    Gets tensorboard log dir
    """
    tb_dir = time.strftime("model_%Y-%m-%d_%H:%M:%S")
    model_dir = "out/tensorboard_logs/{0}/{1}/{2}".format(dataset, model_type, model_note)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    return os.path.join(model_dir, tb_dir) 


def normround(x):
    if x - math.floor(x) < 0.5:
        return math.floor(x)
    return math.ceil(x)


def f1_score(y_true, y_pred):
    """
    F1-score metric taken from old keras source code.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val