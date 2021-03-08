import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from medicalgan.models.gan import GAN
from medicalgan.callbacks.samplegenerator import SampleGenerator
from medicalgan.models.architecture.dcgan_2828 import DCGAN_2828
from medicalgan.utils.utils import get_dataset, get_data_length

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def train():
    """
    Build and train GAN for MNIST image generation.
    """
    rows, cols, channels = (28, 28, 1)
    img_shape = (rows, cols, channels)
    z_dim = 128

    batch_size = 32
    opt = Adam(learning_rate=0.0001)
    loss = BinaryCrossentropy()

    sample_generator = SampleGenerator(z_dim, sample_every=10)
    callbacks = [sample_generator]

    # callbacks = [SampleGenerator(z_dim, sample_every=10)]

    data_dir = "resources/data/mnist/train"
    data = get_dataset(data_dir, rows, cols, batch_size)

    architecture = DCGAN_2828(img_shape, z_dim=z_dim)

    gan = GAN(architecture=architecture, img_shape=img_shape, batch_size=batch_size, z_dim=z_dim)
    gan.compile(gen_opt=opt, dis_opt=opt, loss=loss)

    epochs = 100
    # data_length = get_data_length(data_dir)
    # steps_per_epoch = data_length // batch_size
    workers=4
    max_queue_size=10
    use_multiprocessing=True

    gan.fit(data,
            epochs=epochs,
            # steps_per_epoch=steps_per_epoch,
            workers=workers,
            # max_queue_size=max_queue_size,
            use_multiprocessing=use_multiprocessing,
            callbacks=callbacks)