import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist as mnist

from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Reshape, Flatten, Activation, LeakyReLU
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryCrossentropy, Accuracy, AUC


class GAN():

    def __init__(self, img_shape=(28, 28, 1), z_dim=(100, )):

        self.img_shape = img_shape
        self.z_dim = z_dim

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(),
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        self.discriminator.trainable = False

        self.gan = self.build_gan()

        self.gan.compile(loss='binary_crossentropy',
                         optimizer=Adam())

        self.losses = []
        self.accuracies = []
        self.iteration_checkpoints = []


    def build_generator(self):
        model = Sequential()
        model.add(Dense(128, input_shape=self.z_dim))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        return model


    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1, activation='sigmoid'))
        return model


    def build_gan(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model


    def train(self, iterations, batch_size, sample_interval):

        (X_train, _), (_, _) = mnist.load_data()

        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for iteration in tqdm(range(iterations)):
        
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(z)
        
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
        
            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(z)
            g_loss = self.gan.train_on_batch(z, real)
        
            if (iteration + 1) % sample_interval == 0:
        
                self.losses.append((d_loss, g_loss))
                self.accuracies.append(100.0 * accuracy)
                self.iteration_checkpoints.append(iteration + 1)
        
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                    (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
        
                self.sample_images(self.generator)


    def sample_images(self, image_grid_rows=4, image_grid_columns=4):
        z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, self.z_dim[0]))
        gen_imgs = self.generator.predict(z)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(image_grid_rows,
                                image_grid_columns,
                                figsize=(4, 4),
                                sharey=True,
                                sharex=True)
        cnt = 0
        for i in range(image_grid_rows):
            for j in range(image_grid_columns):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()


mnist_gan = GAN()

iterations = 20000
batch_size = 128
sample_interval = 1000

mnist_gan.train(iterations, batch_size, sample_interval)

#######################

# def build_generator(img_shape, z_dim):
#     model = Sequential()
#     model.add(Dense(128, input_shape=z_dim))
#     model.add(LeakyReLU(alpha=0.01))
#     model.add(Dense(np.prod(img_shape), activation='tanh'))
#     model.add(Reshape(img_shape))
#     return model


# def build_discriminator(img_shape):
#     model = Sequential()
#     model.add(Flatten(input_shape=img_shape))
#     model.add(Dense(128))
#     model.add(LeakyReLU(alpha=0.01))
#     model.add(Dense(1, activation='sigmoid'))
#     return model


# def build_gan(generator, discriminator):
#     model = Sequential()
#     model.add(generator)
#     model.add(discriminator)
#     return model


# def train(iterations, batch_size, sample_interval):
#     (X_train, _), (_, _) = mnist.load_data()
#     X_train = X_train / 127.5 - 1.0
#     X_train = np.expand_dims(X_train, axis=3)
#     real = np.ones((batch_size, 1))
#     fake = np.zeros((batch_size, 1))
#     for iteration in tqdm(range(iterations)):
#         idx = np.random.randint(0, X_train.shape[0], batch_size)
#         imgs = X_train[idx]
#         z = np.random.normal(0, 1, (batch_size, 100))
#         gen_imgs = generator.predict(z)
#         d_loss_real = discriminator.train_on_batch(imgs, real)
#         d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
#         d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
#         z = np.random.normal(0, 1, (batch_size, 100))
#         gen_imgs = generator.predict(z)
#         g_loss = gan.train_on_batch(z, real)
#         if (iteration + 1) % sample_interval == 0:
#             losses.append((d_loss, g_loss))
#             accuracies.append(100.0 * accuracy)
#             iteration_checkpoints.append(iteration + 1)
#             print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
#                   (iteration + 1, d_loss, 100.0 * accuracy, g_loss))
#             sample_images(generator)


# def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
#     z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim[0]))
#     gen_imgs = generator.predict(z)
#     gen_imgs = 0.5 * gen_imgs + 0.5
#     fig, axs = plt.subplots(image_grid_rows,
#                             image_grid_columns,
#                             figsize=(4, 4),
#                             sharey=True,
#                             sharex=True)
#     cnt = 0
#     for i in range(image_grid_rows):
#         for j in range(image_grid_columns):
#             axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
#             axs[i, j].axis('off')
#             cnt += 1
#     plt.show()


######################################

# img_shape = (28, 28, 1)
# z_dim = (100, )

# discriminator = build_discriminator(img_shape)

# discriminator.compile(loss='binary_crossentropy',
#                       optimizer=Adam(),
#                       metrics=['accuracy'])

# generator = build_generator(img_shape, z_dim)

# discriminator.trainable = False

# gan = build_gan(generator, discriminator)

# gan.compile(loss='binary_crossentropy',
#             optimizer=Adam())

# losses = []
# accuracies = []
# iteration_checkpoints = []

# mnist_gan = GAN()

# iterations = 20000
# batch_size = 128
# sample_interval = 1000

# mnist_gan.train(iterations, batch_size, sample_interval)