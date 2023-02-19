import tensorflow
import math

class ImageAnalyzer(tensorflow.keras.layers.Layer):
    def __init__(self, img_width, img_height, channels,n_of_classes,
                number_of_conv_layers=3, depth_of_conv_layer=2,
                decrease_factor=4, learning_rate=2e-4):

        super(ImageAnalyzer, self).__init__()
        self.w, self.h, self.c = img_width, img_height, channels # Size of images
        self.w_b, self.h_b, self.c_b = img_width, img_height, channels
        self.n_conv_layers = number_of_conv_layers 
        self.depth_conv = depth_of_conv_layer
        self.classes = n_of_classes
        self.d_factor = decrease_factor
        self.lr = learning_rate

        # -------- BASE MODEL --------
        # n_conv_layers: Depth of every layer
        # For a depth of 2: Conv2D -> ReLu -> Conv2D -> ReLu
        self.model = tensorflow.keras.Sequential()
        fractional_w = 0.0
        fractional_h = 0.0
        while self.n_conv_layers > 0:

            fractional_w, _ = math.modf(self.w / 2)
            fractional_h, _ = math.modf(self.h / 2)

            if fractional_w != 0 or fractional_h != 0:
                break

            for j in range( self.depth_conv):
                self.model.add(tensorflow.keras.layers.Conv2D(self.c*2,
                                                        (5,5),
                                                        padding = 'same',
                                                        strides = 1,
                                                        activation = 'relu'))
                self.c = self.c*2

            self.model.add(tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)))

            self.w = self.w / 2
            self.h = self.h / 2

            self.n_conv_layers -= 1

        # -------- TOP MODEL --------
        # d_factor: Factor that will make the neurons decrese
        # For a factor of 2: Linear(128, 64) -> Linear(64, 32) ...
        # For a factor of 3: Linear(128, 42) -> Linear(42, 14) ...
        self.model.add(tensorflow.keras.layers.Flatten())
        neurons = int(math.floor(self.w * self.h * self.c))
        while True:
            self.model.add(tensorflow.keras.layers.Dense(int(math.floor(neurons/self.d_factor)),
                                                        activation='relu'))
            neurons = int(math.floor(neurons) / self.d_factor)
            if int(math.floor(neurons) / self.d_factor) <= self.classes:
                self.model.add(tensorflow.keras.layers.Dense(self.classes,
                                                            activation='softmax'))
                break

    def call(self, inputs):
        x = inputs
        for layer in self.model.layers:
            x = layer(x)
        return x

    def compile(self, metrics, loss):
        return self.model.compile(loss=loss, metrics= metrics)
    
    def summary(self):
        self.model.build((None, int(self.w_b), int(self.h_b), int(self.c_b)))
        return self.model.summary()

    def fit(self, x_train, y_train, batch_size, epochs, verbose):
        return self.model.fit(x_train, y_train, batch_size, epochs, verbose)