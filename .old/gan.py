import os
from posix import listdir
import time
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img
 
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

class GAN(Model):
    def __init__(self, img_shape, batch_size=32, z_dim=128, show_summary=False):
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.show_summary = show_summary
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def compile(self, gen_opt, dis_opt, loss):
        super(GAN, self).compile()
        self.gen_opt = gen_opt
        self.dis_opt = dis_opt
        self.loss = loss
        self.gen_loss = Mean(name="gen_loss")
        self.dis_loss = Mean(name="dis_los")

    def train_step(self, X):
        """
        Goal: 
            Train GAN discriminator and generator networks on image batch.
        Args: 
            X: training image batch.
        Steps:
            a) Train discriminator:
                1. generate batch size random latent spaces.
                2. input latent spaces to generator to generate images.
                3. classify batch X and generated images with discriminator.
                4. calculate discriminator loss and update discriminator weights.
            b) Train generator:
                1. generate batch size latent spaces.
                2. input latent spaces to generator to generate images.
                3. classify generated images with discriminator.
                4. calculate generator loss and update generator weights.
        """
        batch_size = tf.shape(X)[0]
        z0 = tf.random.normal(shape=(batch_size, self.z_dim))
        X_gen = self.generator(z0)    
        X_all = tf.concat([X_gen, X], axis=0)
        y_all = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        y_all += 0.05 * tf.random.uniform(tf.shape(y_all))
        with tf.GradientTape() as tape:
            preds = self.discriminator(X_all)
            dis_loss = self.loss(y_all, preds)        
        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        z1 = tf.random.normal(shape=(batch_size, self.z_dim))
        y_gen = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            preds = self.discriminator(self.generator(z1))
            gen_loss = self.loss(y_gen, preds)        
        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        self.dis_loss.update_state(dis_loss)
        self.gen_loss.update_state(gen_loss)
        return {"dis_los": self.dis_loss.result(), "gen_loss": self.gen_loss.result()}

    def build_generator(self):
        model = Sequential(name="generator")
        model.add(Dense(256 * 7 * 7, input_shape=(self.z_dim,)))
        model.add(Reshape((7, 7, 256)))
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
        model.add(Activation('tanh'))
        if self.show_summary:
            model.summary()
        return model

    def build_discriminator(self):
        model = Sequential(name="discriminator")
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        if self.show_summary:
            model.summary()
        return model


class SampleGenerator(Callback):
    def __init__(self, z_dim, n=10, sample_every=1):
        self.n = n
        self.sample_every = sample_every
        self.z_dim = z_dim
        self.out_dir = self.get_out_dir()
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.sample_every == 0:
            z = tf.random.normal(shape=(self.n, self.z_dim))
            X_gen = self.model.generator(z)
            X_gen *= 255
            X_gen.numpy()
            epoch_dir = self.get_epoch_dir(epoch + 1)
            if not os.path.isdir(epoch_dir):
                os.makedirs(epoch_dir)
            for i, gen_img in enumerate(X_gen):
                filename = self.get_out_path(epoch_dir, i)
                img = array_to_img(gen_img)
                img.save(filename)

    def get_out_dir(self):
        out_dir = os.path.join(os.getcwd(), "out/training_samples")
        cur_time = time.strftime("%y-%m-%d_%H:%M:%S")
        return os.path.join(out_dir, cur_time)

    def get_epoch_dir(self, epoch):
        epoch_dir = "epoch_{0}".format(str(epoch))
        return os.path.join(self.out_dir, epoch_dir)

    def get_out_path(self, epoch_dir, i):
        filename = "sample_{0}.jpg".format(str(i))
        return os.path.join(epoch_dir, filename)


def get_dataset(data_dir, rows, cols, batch_size):
<<<<<<< HEAD:mvp/gan.py
    X_train = image_dataset_from_directory(
=======
    data = image_dataset_from_directory(
>>>>>>> 87faaa0270fcfbfe8d0dbefa4a24f9c000bcbf7e:.old/gan.py
        data_dir, 
        label_mode=None, 
        image_size=(rows, cols), 
        color_mode="grayscale",
<<<<<<< HEAD:mvp/gan.py
        batch_size=batch_size
    )
    return X_train.map(lambda x: x / 255.0)
=======
        batch_size=batch_size,
    )
    return data.map(lambda x: x / 255.0)
>>>>>>> 87faaa0270fcfbfe8d0dbefa4a24f9c000bcbf7e:.old/gan.py

def get_data_length(data_dir):
    class_dirs = [os.listdir(os.path.join(data_dir, class_dir)) for class_dir in os.listdir(data_dir)]
    data_length = 0
    for class_dir in class_dirs:
        data_length += len(class_dir)
    return data_length


rows, cols, channels = (28, 28, 1)
img_shape = (rows, cols, channels)
z_dim = 128

batch_size = 32
opt = Adam(learning_rate=0.0001)
loss = BinaryCrossentropy()
callbacks = [SampleGenerator(z_dim, sample_every=10)]

data_dir = "data/train"
data = get_dataset(data_dir, rows, cols, batch_size)

gan = GAN(img_shape=img_shape, batch_size=batch_size, z_dim=z_dim)
gan.compile(gen_opt=opt, dis_opt=opt, loss=loss)

<<<<<<< HEAD:mvp/gan.py
epochs = 20
data_length = get_data_length(data_dir)
steps_per_epoch = data_length // batch_size
=======
epochs = 100
# data_length = get_data_length(data_dir)
# steps_per_epoch = data_length // batch_size
>>>>>>> 87faaa0270fcfbfe8d0dbefa4a24f9c000bcbf7e:.old/gan.py
workers=4
max_queue_size=10
use_multiprocessing=True

<<<<<<< HEAD:mvp/gan.py
print(f"\ndata length: {data_length}\
        \nbatch size: {batch_size}\
        \nsteps per epoch: {steps_per_epoch}\
        \n")

gan.fit(data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        workers=workers,
        max_queue_size=max_queue_size,
=======
gan.fit(data,
        epochs=epochs,
        # steps_per_epoch=steps_per_epoch,
        workers=workers,
        # max_queue_size=max_queue_size,
>>>>>>> 87faaa0270fcfbfe8d0dbefa4a24f9c000bcbf7e:.old/gan.py
        use_multiprocessing=use_multiprocessing,
        callbacks=callbacks)