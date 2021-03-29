from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, LeakyReLU


class VGG():
    """
    CNN model architecture class.
    Classification of binary class images.
    """
    def __init__(self, img_shape, show_summary=False):
        self.img_shape = img_shape
        self.show_summary = show_summary
        self.cnn = self.build_cnn()

    def build_cnn(self):
        """
        VGG-style CNN architecture:

            64 x conv(3,3) -> relu -> 64 x conv(3,3) -> relu 

                -> pool(2,2) -> 128 x conv(3,3) -> relu -> 128 x conv(3,3) -> relu 

                    -> pool(2,2) -> 512 x conv(3,3) -> relu -> 512 x conv(3,3) -> relu -> 512 x conv(3,3) -> relu 

                        -> pool(2, 2) -> 512 x conv(3,3) -> relu -> 512 x conv(3,3) -> relu -> 512 x conv(3,3) -> relu
                        
                            -> pool(2, 2) -> flatten

                                    -> 4096 x dense -> relu -> dropout(0.5) -> 4096 x dense -> relu -> dropout(0.5)

                                        -> 1 x dense -> sigmoid

            (in) - [3,3] - [3,3] -   -                                                                           - O -   - O
            (in) - [3,3] - [3,3] -   -                                                                           - O -   - O
            (in) - [3,3] - [3,3] -   -                                                                           - O -   - O
            (in) - [3,3] - [3,3] -   - [3,3] - [3,3] -   -                                                       - O -   - O
            (in) - [3,3] - [3,3] - p - [3,3] - [3,3] - p -                         p                           p - O - d - O
            (in) - [3,3] - [3,3] - o - [3,3] - [3,3] - o - [3,3] - [3,3] - [3,3] - o -                       - o - O - r - O
            (in) - [3,3] - [3,3] - o - [3,3] - [3,3] - o - [3,3] - [3,3] - [3,3] - o - [3,3] - [3,3] - [3,3] - o - O - o - O - O (out)
            (in) - [3,3] - [3,3] - l - [3,3] - [3,3] - l - [3,3] - [3,3] - [3,3] - l - [3,3] - [3,3] - [3,3] - l - O - p - O 
            (in) - [3,3] - [3,3] - i - [3,3] - [3,3] - i - [3,3] - [3,3] - [3,3] - i -                       - i - O - o - O
            (in) - [3,3] - [3,3] - n - [3,3] - [3,3] - n - [3,3] - [3,3] - [3,3] - n -                       - n - O - u - O
            (in) - [3,3] - [3,3] - g - [3,3] - [3,3] - g -                         g                           g - O - t - O
            (in) - [3,3] - [3,3] -   - [3,3] - [3,3] -   -                                                       - O -   - O
            (in) - [3,3] - [3,3] -   -                                                                           - O -   - O                               
            (in) - [3,3] - [3,3] -   -                                                                           - O -   - O                    
            (in) - [3,3] - [3,3] -   -                                                                           - O -   - O
            (in) - [3,3] - [3,3] -   -                                                                           - O -   - O

        """

        model = Sequential()

        # 2 x 64 * conv(3, 3)
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", input_shape=self.img_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))

        # pooling + 2 x 128 * conv(3, 3)
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))

        # pooling + 3 x 512 * conv(3, 3)
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"))

        # pooling + 3 x 512 * conv(3, 3)
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same"))
        
        # pooling + flatten
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())

        # 2 x dense(4096) + dropout
        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation="relu"))

        # categorisation
        model.add(Dense(1, activation="sigmoid"))

        # summarise
        if self.show_summary:
            model.summary()

        return model
        