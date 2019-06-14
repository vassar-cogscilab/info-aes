#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
# from MADE.KerasImplementation.layers import MaskedDense
from layers import MaskedDense
# from KerasImplementation.layers import MaskedDense
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl

# for now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

# network parameters
batch_size = 100
num_epochs = 500
learning_rate = 0.001  # training parameter
epsilon = 0.000001  # training parameter
hidden_layers = 2
hidden_units = 800
features = 784

# make three-MADE composite network
masks_1 = gen_masks(features, hidden_layers, hidden_units)
masks_2 = gen_masks(features, hidden_layers, hidden_units)
masks_3 = gen_masks(features, hidden_layers, hidden_units)

inputs = tf.keras.Input(shape=(28, 28))
flatten = layers.Flatten()(inputs)  # flatten matrix to vectors

h_1_1 = MaskedDense(hidden_units, masks_1[0], 'relu')(flatten)
h_2_1 = MaskedDense(hidden_units, masks_1[1], 'relu')(h_1_1)
h_out_1 = MaskedDense(features, masks_1[2])(h_2_1)
direct_out_1 = MaskedDense(features, masks_1[3])(flatten)
merge_1 = layers.Add()([h_out_1, direct_out_1])
outputs_1 = layers.Activation('sigmoid')(merge_1)

h_1_2 = MaskedDense(hidden_units, masks_2[0], 'relu')(flatten)
h_2_2 = MaskedDense(hidden_units, masks_2[1], 'relu')(h_1_2)
h_out_2 = MaskedDense(features, masks_2[2])(h_2_2)
direct_out_2 = MaskedDense(features, masks_2[3])(flatten)
merge_2 = layers.Add()([h_out_2, direct_out_2])
outputs_2 = layers.Activation('sigmoid')(merge_2)

h_1_3 = MaskedDense(hidden_units, masks_3[0], 'relu')(flatten)
h_2_3 = MaskedDense(hidden_units, masks_3[1], 'relu')(h_1_3)
h_out_3 = MaskedDense(features, masks_3[2])(h_2_3)
direct_out_3 = MaskedDense(features, masks_3[3])(flatten)
merge_3 = layers.Add()([h_out_3, direct_out_3])
outputs_3 = layers.Activation('sigmoid')(merge_3)

merge = layers.Average()([outputs_1, outputs_2, outputs_3])
unflatten = layers.Reshape((28, 28))(merge)

three_made = Model(inputs=inputs, outputs=unflatten)
three_made.summary()
three_made.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
                   loss='binary_crossentropy')

history = three_made.fit(x=x_train, y=x_train, batch_size=batch_size,
                   epochs=num_epochs, verbose=1)
plt.plot(history.history['loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# save trained network
masks = [masks_1, masks_2, masks_3]
with open('masks_6_14.txt', 'wb') as file:
    pkl.dump(masks, file)


made_weights = three_made.get_weights()
np.savez('weights_6_14', made_weights)

generate_samples(three_made, 25, fname='samples_6_14')