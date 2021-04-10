from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU


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
        
        model.add(Conv2D(100, kernel_size=(3, 3), strides=(10,10),
                        activation='sigmoid', padding ='same',
                        input_shape=self.img_shape))
        
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(50, (3, 3), activation='sigmoid', strides=(5,5), padding ='same'))
        
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(25, kernel_size=(3,3), activation='sigmoid', strides = (1,1), padding ='same'))
        
        model.add(MaxPooling2D(pool_size=(1, 1), padding='valid'))
        model.add(Flatten())
        
        
        model.add(Dense(1, activation='sigmoid'))

        if self.show_summary:
            model.summary()
        return model