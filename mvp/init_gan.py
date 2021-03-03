import os
import time
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist as mnist

from tqdm import tqdm
from tensorflow.keras import Model
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
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        self.generator = self.build_generator()
        self.discriminator.trainable = False
        self.gan = self.build_gan()
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam())
        self.losses = []
        self.accuracies = []
        self.epoch_checkpoints = []
        self.out_dir = get_out_dir()
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

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

    def train(self, epochs=20000, batch_size=128, sample_imgs=True, sample_interval=1000):

        (X_train, _), (_, _) = mnist.load_data()

        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in tqdm(range(epochs)):
        
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
        
            if (epoch + 1) % sample_interval == 0:
        
                self.losses.append((d_loss, g_loss))
                self.accuracies.append(100.0 * accuracy)
                self.epoch_checkpoints.append(epoch + 1)
        
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                    (epoch + 1, d_loss, 100.0 * accuracy, g_loss))
        
                if sample_imgs:
                    self.sample_images(epoch)

    def sample_images(self, epoch, num=10):
        z = np.random.normal(0, 1, (num, self.z_dim[0]))
        gen_imgs = self.generator.predict(z)
        gen_imgs = 225 * (0.5 * gen_imgs + 0.5)
        epoch_dir = get_epoch_dir(self.out_dir, epoch+1)
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)
        for i, gen_img in enumerate(gen_imgs):
            filename = get_out_path(epoch_dir, i)
            cv.imwrite(filename, gen_img)


def get_out_dir():
    out_dir = os.path.join(os.getcwd(), "out")
    cur_time = time.strftime("gan_training_%y-%m-%d_%H:%M:%S")
    return os.path.join(out_dir, cur_time)

def get_epoch_dir(out_dir, epoch):
    epoch_dir = "epoch_{0}".format(str(epoch))
    return os.path.join(out_dir, epoch_dir)

def get_out_path(epoch_dir, i):
    filename = "gen_img_{0}.jpg".format(str(i))
    return os.path.join(epoch_dir, filename)


mnist_gan = GAN()
mnist_gan.train()