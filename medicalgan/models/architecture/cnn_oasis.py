from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU, Dropout


class CNN_OASIS():
    """
    CNN model architecture class.
    Classification of binary class OASIS MRI images.
    """
    def __init__(self, img_shape, show_summary=False):
        self.img_shape = img_shape
        self.show_summary = show_summary
        self.cnn = self.build_cnn()

    def build_cnn(self):

        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=(3, 3), padding ="same", activation="relu", input_shape=self.img_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), padding ="same", activation="relu"))
        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2), strides=(2,2)))
        model.add(Conv2D(64, kernel_size=(3, 3), padding ="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), padding ="same", activation="relu"))
        model.add(BatchNormalization())

        # model.add(MaxPooling2D((2, 2), strides=(2,2)))
        # model.add(Conv2D(128, kernel_size=(3, 3), padding ="same", activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Conv2D(128, kernel_size=(3, 3), padding ="same", activation="relu"))
        # model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2), strides=(2,2)))
        model.add(Flatten())

        model.add(Dense(128, activation="relu"))    
        model.add(Dropout(0.7))
        model.add(Dense(64, activation="relu"))
        
        model.add(Dense(1, activation='sigmoid'))

        if self.show_summary:
            model.summary()
        return model