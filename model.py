from keras.layers import Conv2D, Dense, Dropout, Activation, MaxPooling2D
from keras.models import Sequential
import pickle

# Create model
model = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding="same",
           data_format="channels_last"),
    Activation('relu'),
    MaxPooling2D(strides=(2, 2)),
    Conv2D(64, kernel_size=5, strides=1),
    Activation('relu'),
    MaxPooling2D(strides=(2, 2)),
    Dense(1024),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
])

model.compile(optimizer="rmsprop",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load data
with open('./cifar-10-batches-py/data_batch_1', 'rb') as d1:
    data = pickle.load(d1, encoding='bytes')

# Train model
model.fit(data, epochs=2, validation_split=0.8)

# Evaluate model

# Print results
