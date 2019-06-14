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
import generators

# for now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

# network parameters
batch_size = 100
num_epochs = 900
learning_rate = 0.001  # training parameter
epsilon = 0.000001  # training parameter
hidden_layers = 2
hidden_units = 2000
features = 784
num_masks = 10
first_training = False  # set to True to train a new model
per_batch_training = False  # set to True to use train_on_batch
if num_masks > 1:
    per_batch_training = True

if first_training:
    masks = generators.gen_masks(num_masks, features, hidden_layers, hidden_units)

else:
    with open('masks_2000_made.txt', 'rb') as file:
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

if first_training == False:
    # ensure that network is properly initialized
    made.train_on_batch(x=x_train[0:batch_size], y=x_train[0:batch_size])

    # load network weights from test file
    weights = np.load('weights_2000_made.npz', allow_pickle=True)['arr_0']
    made.set_weights(weights)

if per_batch_training:
    # train with mini-batch mask randomization
    history = []
    for i in range(num_epochs):
        # prepare batched data to feed to network
        data = []
        while len(data) < batches_per_epoch:
            print('Batching data for epoch ' + str(i)+ 'of ' + str(num_epochs) + '.')
            np.random.shuffle(x_train)
            if len(data) * batch_size + batch_size > x_train.shape[0]:
                batch = x_train[len(data) * batch_size:]
            else:
                batch = x_train[len(data) * batch_size:(len(data) * batch_size) +
                        batch_size]
            data.append(batch)
        print('Epoch ' + str(i) + ' of ' + str(num_epochs) + '.')
        for j in range(batches_per_epoch):
            if j % 100 == 0:
                print('Batch ' + str(j) + ' of ' + str(batches_per_epoch) + '.')
            batches_completed = (batches_per_epoch * i) + j
            next_mask_set = batches_completed % len(masks) + 1
            # train on one batch and add the loss to the history
            history.append(made.train_on_batch(x=data[j], y=data[j]))
            # take weights from MADE
            weights = made.get_weights()
            tf.keras.backend.clear_session()
            # remake network with next mask set
            inputs = tf.keras.Input(shape=(28, 28))
            flatten = tf.keras.layers.Flatten()(inputs)
            h_1 = MaskedDense(hidden_units, masks[next_mask_set][0], 'relu')(flatten)
            h_2 = MaskedDense(hidden_units, masks[next_mask_set][1], 'relu')(h_1)
            h_out = MaskedDense(784, masks[next_mask_set][2])(h_2)
            direct_out = MaskedDense(784, masks[next_mask_set][3])(flatten)
            merge = tf.keras.layers.Add()([h_out, direct_out])
            outputs = tf.keras.layers.Activation('sigmoid')(merge)
            unflatten = tf.keras.layers.Reshape((28, 28))(outputs)
            made = Model(inputs=inputs, outputs=unflatten)
            # restore previous training
            made.set_weights(weights)
            made.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
                        loss='binary_crossentropy')
    # Plot training loss values
    plt.plot(history)
    plt.title('Model ' + str(i) + ' loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.show()

else:
    history = made.fit(x=x_train, y=x_train, batch_size=batch_size,
                    epochs=num_epochs, verbose=1)
    plt.plot(history.history['loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

# save trained network
with open('masks_2000_made_6_14.txt', 'wb') as file:
    pkl.dump(masks, file)

made_2000_weights = made.get_weights()
np.savez('weights_2000_made_6_14', made_2000_weights)

generators.generate_samples(made, 25, 'samples_2000_6_14')