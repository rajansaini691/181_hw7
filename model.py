from tensorflow.keras.layers import Conv2D, Dense, Dropout, Activation
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set to True to force retraining
retrain = True

# Create model
model = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding="same",
           data_format="channels_last", input_shape=(32, 32, 3)),
    Activation('relu'),
    MaxPooling2D(strides=(2, 2)),
    Conv2D(64, kernel_size=5, strides=1),
    Activation('relu'),
    MaxPooling2D(strides=(2, 2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax', name='final_activation')
])

rms = RMSprop(learning_rate=0.0005)
model.compile(optimizer=rms,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

if retrain:
    # Train model
    history = model.fit(x_train, y_train, epochs=8, validation_split=0.2)

    # Save model
    model.save("./model.h5")

    # Plot accuracy over time
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot loss over time
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
else:
    model.load_weights("./model.h5")

# Evaluate model
scores = model.evaluate(x_test, y_test, verbose=1)

# Print number of parameters
print(f'Parameters: {model.count_params()}')

# Save misclassified and correctly-classified samples
classes = model.predict_classes(x_test)
misclassified = [(0, 0)]*10     # index, predicted
correct = [(0, 0)]*10           # index, class
for i in range(len(classes)):
    predicted = classes[i]
    actual = y_test.argmax(axis=1)[i]
    if predicted - actual != 0:
        misclassified[actual] = (i, predicted)
    else:
        correct[predicted] = (i, predicted)

for i, x in enumerate(misclassified):
    idx = x[0]
    pred = x[1]
    print((x_test[idx]*256).astype(np.uint8))
    img = Image.fromarray((x_test[idx]*256).astype(np.uint8))

    # Saved as {predicted}_{actual}.png
    img.save(f"./images/misclassified/{pred}_{i}.png")

for idx, pred in correct:
    img = Image.fromarray((x_test[idx]*256).astype(np.uint8))
    img.save(f"./images/correct/{pred}.png")


# Print results
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
