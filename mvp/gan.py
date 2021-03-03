import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, MaxPool2D, Activation, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img


class GAN(Model):

    def __init__(self, img_shape, z_dim):
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

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
        model.summary()
        return model

    def compile(self, generator_opt, discriminator_opt, loss_func):
        super(GAN, self).compile()
        self.generator_opt = generator_opt
        self.discriminator_opt = discriminator_opt
        self.loss_func = loss_func
        self.generator_loss_func = Mean(name="generator_loss")
        self.discriminator_loss_func = Mean(name="discriminator_loss")

    def train_step(self, real_images):

        # generate random latent space for generator
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.z_dim))

        # generate images from latent space
        generated_images = self.generator(random_latent_vectors)

        # combine generated images with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # create labels for generated and real images and add random noise
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # get discriminator loss with gradient tape 
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_func(labels, predictions)

        # apply gradients to discriminator
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # generate random latent space for generator and labels
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.z_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        # get generator loss with gradient tape
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_func(misleading_labels, predictions)

        # apply gradients to generator
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_opt.apply_gradients(zip(grads, self.generator.trainable_weights))

        # update loss metrics
        self.discriminator_loss_func.update_state(d_loss)
        self.generator_loss_func.update_state(g_loss)
        return {"discriminator_loss": self.discriminator_loss_func.result(),
                "generator_loss": self.generator_loss_func.result()}


class SampleGAN(Callback):

    def __init__(self, z_dim, n=10):
        self.n = n
        self.z_dim = z_dim
        self.out_dir = self.get_out_dir()
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.n, self.z_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        epoch_dir = self.get_epoch_dir(epoch)
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)
        for i, gen_img in enumerate(generated_images):
            filename = self.get_out_path(epoch_dir, i)
            img = array_to_img(gen_img)
            img.save(filename)

    def get_out_dir(self):
        out_dir = os.path.join(os.getcwd(), "out")
        cur_time = time.strftime("gan_training_%y-%m-%d_%H:%M:%S")
        return os.path.join(out_dir, cur_time)

    def get_epoch_dir(self, epoch):
        epoch_dir = "epoch_{0}".format(str(epoch))
        return os.path.join(self.out_dir, epoch_dir)

    def get_out_path(self, epoch_dir, i):
        filename = "gen_img_{0}.jpg".format(str(i))
        return os.path.join(epoch_dir, filename)


rows, cols, channels = (28, 28, 1)
img_shape = (rows, cols, channels)
batch_size = 32

dataset = image_dataset_from_directory("data/train", 
                                       label_mode=None, 
                                       image_size=(rows, cols), 
                                       color_mode="grayscale",
                                       batch_size=batch_size)

dataset = dataset.map(lambda x: x / 255.0)

gan = GAN(img_shape=img_shape, z_dim=100)

gan.compile(generator_opt=Adam(learning_rate=0.0001), 
            discriminator_opt=Adam(learning_rate=0.0001),
            loss_func=BinaryCrossentropy())

gan.fit(dataset, 
        epochs=10, 
        callbacks=[SampleGAN(gan.z_dim)])