#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from MADE.KerasImplementation.layers import MaskedDense
# from layers import MaskedDense
# from KerasImplementation.layers import MaskedDense
import numpy as np
import matplotlib.pyplot as plt
import math

# for now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

# network parameters
batch_size = 100
num_epochs = 100
learning_rate = 0.001  # training parameter
epsilon = 0.000001  # training parameter
hidden_layers = 2
hidden_units = 500
features = 784


# generate an array of mask matrices
def gen_masks(features, hidden_layers, hidden_units):

    first_input_indices = []
    for j in range(features):
        first_input_indices.append(j+1)
    input_indices = []
    input_dim = hidden_units
    masks = []
    input_indices = []
    # generate a mask for each hidden layer
    for i in range(hidden_layers):
        if len(input_indices) == 0:
            input_dim = features
            input_indices = first_input_indices
            np.random.shuffle(input_indices)
        # we must generate a list of node indices for this layer
        layer_indices = []
        for j in range(hidden_units):
            layer_indices.append(np.random.randint(min(input_indices), features - 1))
        mask = np.zeros((input_dim, hidden_units), dtype=np.float32)
        # populate mask with appropriate values
        for j in range(len(input_indices)): # iterate over every layer node
            for k in range(len(layer_indices)): # iterate over every input node
                if input_indices[j] <= layer_indices[k]:
                    mask[j][k] = 1
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        masks.append(mask)
        input_indices = layer_indices
        input_dim = len(input_indices)
    # generate output layer masks
    output_mask = np.zeros((hidden_units, features), dtype=np.float32)
    for j in range(len(input_indices)):  # every layer node
        for k in range(len(first_input_indices)):  # every input node
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
    return masks

# visualize auto-encoding and generative capacities


def auto_encode(model, num_samples, fname=None):

    # visualize inputs
    original_images = plt.figure(figsize=(5, 5), facecolor='#ffffff')
    # visualize auto-encodings
    output_images = plt.figure(figsize=(5, 5), facecolor='#ffffff')
    for i in range(num_samples):
        proto_input = x_test[np.random.randint(0, x_test.shape[0]+1)]
        subplot = original_images.add_subplot(math.sqrt(num_samples),
                                              math.sqrt(num_samples), i+1)
        subplot.imshow(proto_input, cmap='gray')
        subplot.axis('off')
        input = np.empty((1, 28, 28))
        input[0] = proto_input
        output = model.predict(input, batch_size=1)
        subplot = output_images.add_subplot(math.sqrt(num_samples),
                                            math.sqrt(num_samples), i+1)
        subplot.imshow(output[0], cmap='gray')
        subplot.axis('off')
    if fname is None:
        plt.show()
    if fname is not None:
        plt.savefig(fname)


def generate_samples(model, num_samples, fname=None):

    noise_parameter = np.random.rand()
    plot_size = math.ceil(math.sqrt(num_samples))
    generated_samples = plt.figure(figsize=(10, 10), facecolor='#ffffff')
    for i in range(num_samples):
        noise = np.random.binomial(1, noise_parameter, size=(1, 28, 28))
        output = np.zeros(noise[0].shape, dtype=np.float32)
        row_length = noise.shape[1]
        for j in range(1, len(noise.flatten())): 
            noise = model.predict(noise, batch_size=1)
            p = noise[0][j // row_length][j % row_length]
            sample = np.random.binomial(1, p)
            noise[0][j // row_length][j % row_length] = p
            output[j // row_length][j % row_length] = sample
        subplot = generated_samples.add_subplot(plot_size, plot_size, i+1)
        subplot.imshow(output, cmap='gray')
        subplot.axis('off')
    if fname is None:
        plt.show()
    if fname is not None:
        plt.savefig(fname)

# make three-MADE composite network
masks_1 = gen_masks(features, hidden_layers, hidden_units)
masks_2 = gen_masks(features, hidden_layers, hidden_units)
masks_3 = gen_masks(features, hidden_layers, hidden_units)

inputs = tf.keras.Input(shape=(28, 28))
flatten = tf.keras.layers.Flatten()(inputs)  # flatten matrix to vectors

h_1_1 = MaskedDense(hidden_units, masks_1[0], 'relu')(flatten)
h_2_1 = MaskedDense(hidden_units, masks_1[1], 'relu')(h_1_1)
h_out_1 = MaskedDense(features, masks_1[2])(h_2_1)
direct_out_1 = MaskedDense(features, masks_1[3])(flatten_1)
merge_1 = tf.keras.layers.Add()([h_out_1, direct_out_1])
outputs_1 = tf.keras.layers.Activation('sigmoid')(merge_1)

h_1_2 = MaskedDense(hidden_units, masks_2[0], 'relu')(flatten)
h_2_2 = MaskedDense(hidden_units, masks_2[1], 'relu')(h_1_2)
h_out_2 = MaskedDense(features, masks_2[2])(h_2_2)
direct_out_2 = MaskedDense(features, masks_2[3])(flatten_2)
merge_2 = tf.keras.layers.Add()([h_out_2, direct_out_2])
outputs_2 = tf.keras.layers.Activation('sigmoid')(merge_2)

h_1_3 = MaskedDense(hidden_units, masks_3[0], 'relu')(flatten)
h_2_3 = MaskedDense(hidden_units, masks_3[1], 'relu')(h_1_3)
h_out_3 = MaskedDense(features, masks_3[2])(h_2_3)
direct_out_3 = MaskedDense(features, masks_3[3])(flatten_3)
merge_3 = tf.keras.layers.Add()([h_out_3, direct_out_3])
outputs_3 = tf.keras.layers.Activation('sigmoid')(merge_3)

average = tf.keras.layers.Average([outputs_1, outputs_2, outputs_3])
unflatten = tf.keras.layers.Reshape((28, 28))(average)

three_made = Model(inputs=inputs, outputs=unflatten)
three_made.summary()
three_made.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
                   loss='binary_crossentropy')

history = made.fit(x=x_train, y=x_train, batch_size=batch_size,
                   epochs=num_epochs, verbose=1)
plt.plot(history['history'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()

generate_samples(made, 25, 'samples')