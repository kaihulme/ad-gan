import os
from tensorflow.keras.preprocessing import image_dataset_from_directory


def get_dataset(data_dir, rows, cols, batch_size):
    """
    Creates tf.data object from directory.
    """
    target_dir = os.path.join(os.getcwd(), data_dir)
    data = image_dataset_from_directory(
        target_dir, 
        label_mode=None, 
        image_size=(rows, cols), 
        color_mode="grayscale",
        batch_size=batch_size,
    )
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