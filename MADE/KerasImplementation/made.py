#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from MADE.KerasImplementation.layers import MaskedDense
# from KerasImplementation.layers import MaskedDense
# from layers import MaskedDense
import numpy as np
import matplotlib.pyplot as plt
import math
import generators

# for now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

# network parameters
batch_size = 100
num_epochs = 1
learning_rate = 0.01
epsilon = 0.000001
num_masks = 10
hidden_layers = 2
hidden_units = 1000
features = 784

batches_per_epoch = math.ceil(x_train.shape[0] / batch_size)
remainder_batch_size = x_train.shape[0] - ((batches_per_epoch-1) * batch_size)
total_batches = batches_per_epoch * num_epochs

# generate an array of mask matrices
def gen_masks(num_masks, features, hidden_layers, hidden_units, 
              classes=None):

    all_masks = []
    if classes is not None:
        label_indices = []
        for i in range(classes):
            label_indices.append(0)
    first_input_indices = []
    for i in range(features):
        first_input_indices.append(i+1)
    for m in range(num_masks):
        masks = []
        input_indices = []
        # generate a mask for each hidden layer
        for i in range(hidden_layers):
            if len(input_indices) == 0:
                input_indices = first_input_indices
                np.random.shuffle(input_indices)
                if classes is not None:
                    input_indices = label_indices + input_indices
            # we must generate a list of node indices for this layer
            layer_indices = []
            for j in range(hidden_units):
                layer_indices.append(np.random.randint(min(input_indices),
                                     features - 1))
            mask = np.zeros((len(input_indices), len(layer_indices)), 
                            dtype=np.float32)
            # populate mask with appropriate values
            for j in range(len(input_indices)):  # every layer node
                for k in range(len(layer_indices)):  # every input node
                    if input_indices[j] <= layer_indices[k]:
                        mask[j][k] = 1
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            masks.append(mask)
            input_indices = layer_indices
        # generate output layer masks
        output_mask = np.zeros((hidden_units, features), dtype=np.float32)
        for j in range(len(input_indices)):
            for k in range(len(first_input_indices)):
                if input_indices[j] < first_input_indices[k]:
                    output_mask[j][k] = 1
        output_mask = tf.convert_to_tensor(output_mask, dtype=tf.float32)
        masks.append(output_mask)
        direct_mask = np.zeros((features, features), dtype=np.float32)
        for j in range(len(first_input_indices)):
            for k in range(len(first_input_indices)):
                if first_input_indices[j] <= first_input_indices[k]:
                    direct_mask[j][k] = 1
        direct_mask = tf.convert_to_tensor(direct_mask, dtype=tf.float32)
        masks.append(direct_mask)
        all_masks.append(masks)
    return all_masks

# generate set of masks
masks = gen_masks(num_masks, features, hidden_layers, hidden_units)

# make network
inputs = tf.keras.Input(shape=(28, 28))
flatten = tf.keras.layers.Flatten()(inputs)  # flatten matrix data to vectors
h_1 = MaskedDense(hidden_units, masks[0][0], 'relu')(flatten)
h_2 = MaskedDense(hidden_units, masks[0][1], 'relu')(h_1)
h_out = MaskedDense(784, masks[0][2])(h_2)
direct_out = MaskedDense(784, masks[0][3])(flatten)
merge = tf.keras.layers.Add()([h_out, direct_out])
outputs = tf.keras.layers.Activation('sigmoid')(merge)
unflatten = tf.keras.layers.Reshape((28, 28))(outputs)
made = Model(inputs=inputs, outputs=unflatten)
made.summary()
made.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
             loss='binary_crossentropy')

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