#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from layers import MaskedDense
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl
import generators

# Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

# Hyperparameters
batch_size = 100
num_epochs = 900
learning_rate = 0.001  # training parameter
epsilon = 0.000001  # training parameter
hidden_layers = 2
hidden_units = 2000
features = 784

# Load trained network
with open('masks_2000_made_6_14.txt', 'rb') as file:
    masks = pkl.load(file)

# make network
inputs = tf.keras.Input(shape=(28, 28))
flatten = tf.keras.layers.Flatten()(inputs)  # flatten matrix to vectors
h_1 = MaskedDense(hidden_units, masks[0], 'relu')(flatten)
h_2 = MaskedDense(hidden_units, masks[1], 'relu')(h_1)
h_out = MaskedDense(features, masks[2])(h_2)
direct_out = MaskedDense(features, masks[3])(flatten)
merge = tf.keras.layers.Add()([h_out, direct_out])
outputs = tf.keras.layers.Activation('sigmoid')(merge)
unflatten = tf.keras.layers.Reshape((28, 28))(outputs)
made = Model(inputs=inputs, outputs=unflatten)
made.summary()
made.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
             loss='binary_crossentropy')

# ensure that network is properly initialized
made.train_on_batch(x=x_train[0:batch_size], y=x_train[0:batch_size])

# load network weights from test file
weights = np.load('weights_2000_made_6_14.npz', allow_pickle=True)['arr_0']
made.set_weights(weights)

images = []
for i in range(100):
    choice = np.random.choice(range(x_test.shape[0]))
    images.append(x_test[choice])

generators.info_reorder(made, images, 'info_order_200_made_6_14')