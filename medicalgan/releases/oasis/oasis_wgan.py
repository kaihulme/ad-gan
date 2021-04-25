import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from medicalgan.models.wgan import WGAN
from medicalgan.callbacks.samplegenerator import SampleGenerator
from medicalgan.models.architecture.dcgan_oasis import DCGAN_OASIS
from medicalgan.utils.utils import get_dataset, get_singleclass_aug_dataset, get_singleclass_data, get_model_path, get_tb_dir

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

DATASET = "oasis"
MODEL_TYPE = "wgan"
MODEL_NOTE = "176_176"


def train(plane, depth, label):
	"""
	Build and train GAN for MNIST image generation.
	"""
	rows, cols, channels = (176, 176, 1)
	img_shape = (rows, cols, channels)
	z_dim = 128

	batch_size = 6
	opt = Adam(learning_rate=0.0001)
	loss = BinaryCrossentropy()

	sample_generator = SampleGenerator(DATASET, MODEL_TYPE, MODEL_NOTE, z_dim, sample_every=10, label=label)
	tensorboard = TensorBoard(get_tb_dir(DATASET, MODEL_TYPE, MODEL_NOTE))
	callbacks = [sample_generator, tensorboard]

	data_dir = "resources/data/oasis/{0}/{1}/train".format(depth, plane)
	data = get_dataset(data_dir, rows, cols, batch_size, pixrange="-11")

	architecture = DCGAN_OASIS(img_shape, z_dim=z_dim)

	wgan = WGAN(architecture=architecture, img_shape=img_shape, batch_size=batch_size, z_dim=z_dim, n_critics=1)
	wgan.compile(gen_opt=opt, dis_opt=opt, loss=loss)

	epochs = 100
	workers=4
	max_queue_size=10
	use_multiprocessing=True

	wgan.fit(
		data,
		batch_size=batch_size,
		epochs=epochs,
		workers=workers,
		max_queue_size=max_queue_size,
		use_multiprocessing=use_multiprocessing,
		callbacks=callbacks
	)

	wgan_path = get_model_path(DATASET, MODEL_TYPE, MODEL_NOTE, label)
	tf.saved_model.save(wgan, wgan_path)