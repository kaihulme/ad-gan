import os
import time
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img


class SampleGenerator(Callback):
    """
    Outputs generated images every specified epoch.
    """
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