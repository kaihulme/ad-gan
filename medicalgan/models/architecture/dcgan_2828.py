from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU


class DCGAN_2828():
    """
    DCGAN model architecture class.
    Generator generates 28*28 images.
    Discriminator takes two 28*28 images.
    """
    def __init__(self, img_shape, z_dim=128, show_summary=False):
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.show_summary = show_summary
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
        if self.show_summary:
            model.summary()
        return model

    def build_discriminator(self):
        model = Sequential(name="discriminator")
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        if self.show_summary:
            model.summary()
        return model