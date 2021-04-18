import os
import math
import time
from tensorflow.keras.preprocessing import image_dataset_from_directory
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