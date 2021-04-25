import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from medicalgan.models.gan import GAN
from medicalgan.callbacks.samplegenerator import SampleGenerator
from medicalgan.models.architecture.dcgan_2828 import DCGAN_2828
from medicalgan.utils.utils import get_dataset, get_tb_dir, get_model_path

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

DATASET = "mnist"
MODEL_TYPE = "small_gan"
MODEL_NOTE = "28"


def train():
    """
    Build and train GAN for MNIST image generation.
    """
    rows, cols, channels = (28, 28, 1)
    img_shape = (rows, cols, channels)
    z_dim = 128

    batch_size = 64
    opt = Adam(learning_rate=0.0001)
    loss = BinaryCrossentropy()

    sample_generator = SampleGenerator(DATASET, MODEL_TYPE, MODEL_NOTE, z_dim, sample_every=1)
    tensorboard = TensorBoard(get_tb_dir(DATASET, MODEL_TYPE, MODEL_NOTE))
    callbacks = [sample_generator, tensorboard]

    data_dir = "resources/data/mnist/train/"
    data = get_dataset(data_dir, rows, cols, batch_size)

    for x in data:
        print(x.shape)
        break

    architecture = DCGAN_2828(img_shape, z_dim=z_dim)

    gan = GAN(architecture=architecture, img_shape=img_shape, batch_size=batch_size, z_dim=z_dim)
    gan.compile(gen_opt=opt, dis_opt=opt, loss=loss)

    epochs = 100
    workers=4
    max_queue_size=10
    use_multiprocessing=True

    gan.fit(data,
            epochs=epochs,
            workers=workers,
            max_queue_size=max_queue_size,
            use_multiprocessing=use_multiprocessing,
            callbacks=callbacks
    )

    gan_path = get_model_path(DATASET, MODEL_TYPE, MODEL_NOTE)
    tf.saved_model.save(gan, gan_path)