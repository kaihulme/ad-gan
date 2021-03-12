from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU


class CNN_64():
    """
    CNN model architecture class.
    Classification of binary class 64x64 images.
    """
    def __init__(self, img_shape, show_summary=False):
        self.img_shape = img_shape
        self.show_summary = show_summary
        self.cnn = self.build_cnn()

    def build_cnn(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=self.img_shape))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        if self.show_summary:
            model.summary()
        return model