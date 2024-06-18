# @title **Image Classification using CNN**

# @markdown This project demonstrates how to build and train a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras.

# @markdown ## Steps:
# @markdown 1. Import Libraries
# @markdown 2. Load Dataset
# @markdown 3. Preprocess Data
# @markdown 4. Build CNN Model
# @markdown 5. Compile and Train Model
# @markdown 6. Evaluate Model
# @markdown 7. Make Predictions

# ## 1. Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10

# ## 2. Load Dataset
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to a pixel value range of 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Print the shape of the dataset
print(f'Training data shape: {x_train.shape}')
print(f'Test data shape: {x_test.shape}')

# ## 3. Preprocess Data
# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Display some sample images from the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(y_train[i])])
plt.show()

# ## 4. Build CNN Model
# Initialize the model
model = Sequential()

# Add convolutional layers, pooling layers, and dense layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# ## 5. Compile and Train Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# ## 6. Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# ## 7. Make Predictions
# Predict the first 5 images in the test set
predictions = model.predict(x_test[:5])

# Display predictions with the actual images
plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(predictions[i])])
plt.show()
