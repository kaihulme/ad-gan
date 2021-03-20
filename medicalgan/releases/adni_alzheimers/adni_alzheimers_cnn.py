import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from medicalgan.models.architecture.cnn_64 import CNN_64
from medicalgan.utils.utils import get_dataset, get_data_length

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def train():
    """
    Build and train CNN for tumour detection in fMRI images.
    """
    rows, cols, channels = (150, 150, 1)
    img_shape = (rows, cols, channels)

    batch_size = 32
    opt = Adam(learning_rate=0.0001)
    loss = BinaryCrossentropy()
    metrics = ["accuracy"]

    # callbacks = []

    train_dir = "resources/data/adni_alzheimers/train"
    val_dir = "resources/data/adni_alzheimers/val"
    # test_dir = "resources/data/adni_alzheimers/test"

    train_data = get_dataset(train_dir, rows, cols, batch_size, label_mode="binary")
    val_data = get_dataset(val_dir, rows, cols, batch_size, label_mode="binary")
    # test_data = get_dataset(test_dir, rows, cols, batch_size, label_mode="binary")

    architecture = CNN_64(img_shape)

    cnn = architecture.cnn
    cnn.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics,
    )

    epochs = 100
    # data_length = get_data_length(data_dir)
    # steps_per_epoch = data_length // batch_size
    # workers=4
    # max_queue_size=10
    # use_multiprocessing=True

    cnn.fit(
        train_data,
        validation_data = val_data,
        epochs=epochs,
        # steps_per_epoch=steps_per_epoch,
        # workers=workers,
        # max_queue_size=max_queue_size,
        # use_multiprocessing=use_multiprocessing,
        # callbacks=callbacks,
    )