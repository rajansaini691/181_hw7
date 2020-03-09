from keras.layers import Conv2D, Dense, Activation
from keras.models import Sequential

model = Sequential([
    Conv2D(32, 5, 1, padding="same", data_format="channels_last"),
    Activation('relu'),
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
