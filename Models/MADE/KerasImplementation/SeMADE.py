#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl
import os

cwd = os.getcwd()
cwd = cwd + '/MADE/KerasImplementation'
os.chdir(cwd)

from layers import MaskedDense
import generators

# For now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(y_train.shape)

# Binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

# Do one-hot encoding of data labels
y_train = K.one_hot(y_train, 10)
y_test = K.one_hot(y_test, 10)

# Network parameters
batch_size = 100
num_epochs = 900
learning_rate = 0.001  # training parameter
epsilon = 0.000001  # training parameter
hidden_layers = 2
hidden_units = 2000
features = 784
num_masks = 1
classes = 10

masks = generators.gen_masks(num_masks, features, hidden_layers, hidden_units, 10)

image_inputs = tf.keras.Input(shape=(28, 28))
class_inputs = tf.keras.Input(shape=(10,))
flatten = layers.Flatten()(image_inputs)
inputs = layers.Concatenate(axis=1)([class_inputs, flatten])
h_1 = MaskedDense(hidden_units, masks[0][0], 'relu')(inputs)
h_2 = MaskedDense(hidden_units, masks[0][1], 'relu')(h_1)
h_out = MaskedDense(features, masks[0][2])(h_2)
direct_out = MaskedDense(features, masks[0][3])(inputs)
merge = tf.keras.layers.Add()([h_out, direct_out])
unflatten = tf.keras.layers.Reshape((28, 28))(outputs)
SeMADE = Model(inputs=inputs, outputs=unflatten)
SeMADE.summary()
SeMADE.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
             loss='binary_crossentropy')

history = SeMADE.fit(x=x_train, y=x_train, batch_size=batch_size,
                   epochs=num_epochs, verbose=1)
plt.plot(history.history['loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Save network after training
with open('semade_masks.txt', 'wb') as file:
    pkl.dump(masks, file)

made_2000_weights = made.get_weights()
np.savez('semade_weights', made_2000_weights)

generate_samples(made, 25, 'semade_samples')