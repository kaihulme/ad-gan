import os
from tensorflow.keras.preprocessing import image_dataset_from_directory


def normalise_dataset(x, y):
    return x / 255.0, y


def get_dataset(data_dir, rows, cols, batch_size, label_mode=None, color_mode="grayscale"):
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