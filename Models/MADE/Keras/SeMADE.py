#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl
from layers import MaskedDense
from layers import AddWithBias
import generators
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '[3, 4]'


def gen_masks(num_masks, features, classes, hidden_layers, hidden_units):

    """
    Creates a desired number of sets of masks for SeMADE.

    Arguments:
        num_masks: an int equal to the number of different sets of masks
            desired.
        features: an int equal to the number of features of the data.
        classes: an int equal to the number of categories the data points fall
            into.
        hidden_layers: an int equal to the number of hidden layers in MADE.
        hidden_units: an int equal to the number of hidden units in MADE's
            hidden layers.

    Returns:
        A dictionary with keys 'masks' and 'indices'. Under key 'masks',
        there is a list containing num_masks lists of different masks for
        SeMADE. The masks in these lists are in the order: hidden layer 1,
        hidden layer 2, output, and direct connection. Under key 'indices',
        there is a list containing the list of input/output node indices
        for each of the mask sets generated. The indices of the two lists
        in the dictionary correspond--the first mask set goes with the first
        set of indices, and so on.
    """

    # SeMADE has two inputs--one for images and one for one-hot encoded
    #   label vectors. These indices are for the concatenated input layer:
    #   labels followed by flattened images.
    label_indices = []
    input_indices = []
    for i in range(classes):
        # The entire class vector needs to connect to the hidden layers,
        #   so all nodes corresponding to its elements will have index = 0.
        label_indices.append(0)
    for i in range(features):
        # The nodes corresponding to the dimensions of the input will have
        #   indices from 1 to features.
        # The iterator starts at 0, the indices start at 1
        input_indices.append(i + 1)
    masks = []
    indices = []  # Record input layer indices for sampling
    for i in range(num_masks):
        set_masks = []
        set_image_indices = input_indices
        np.random.shuffle(set_image_indices)  # Random ordering of image pixels
        set_inputs = label_indices + set_image_indices
        prev_indices = set_inputs
        for j in range(hidden_layers):
            layer_indices = []
            for k in range(hidden_units):
                layer_indices.append(np.random.randint(low=min(prev_indices),
                                                       high=features))
            mask = np.zeros((len(prev_indices), len(layer_indices)),
                            dtype=np.float32)
            for k in range(len(prev_indices)):
                for l in range(len(layer_indices)):
                    # Always allow connection bewteen the label vector and the
                    #   next layer.
                    if prev_indices[k] == 0:
                        mask[k][l] = 1
                    # The mask value will be one when the autoregressive
                    #   condition is met.
                    else:
                        mask[k][l] = float(int(prev_indices[k] <= layer_indices[l]))
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            set_masks.append(mask)
            prev_indices = layer_indices
        output_mask = np.zeros((len(prev_indices), features), dtype=np.float32)
        for j in range(len(prev_indices)):
            for k in range(len(set_image_indices)):
                output_mask[j][k] = float(int(prev_indices[j] < set_image_indices[k]))
        output_mask = tf.convert_to_tensor(output_mask, dtype=tf.float32)
        set_masks.append(output_mask)
        direct_mask = np.zeros((features + classes, features),
                               dtype=np.float32)
        for j in range(features + classes):
            for k in range(features):
                if set_inputs[j] == 0:
                    direct_mask[j][k] = 1
                else:
                    direct_mask[j][k] = float(int(set_inputs[j] < set_image_indices[k]))
        direct_mask = tf.convert_to_tensor(direct_mask, dtype=tf.float32)
        set_masks.append(direct_mask)
        masks.append(set_masks)
        indices.append([set_inputs])
    return{'masks': masks, 'indices': indices}


def logx_loss(y_true, y_pred):
    """
    Computes the negative log-likelihood of an image outputted by MADE.

    Arguments:
        y_true: the tensor inputted to the network.
        y_pred: the tensor computed by the network.
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    xent_loss = 28 * 28 * tf.keras.metrics.binary_crossentropy(y_true_flat,
                                                               y_pred_flat)
    return xent_loss


def batch_data(images, labels, batch_size, batches_per_epoch):
    """ 
    Produces a list of batches for one epoch from a training dataset.

    Arguments:
        data: an array or tensor containing all the training data.
        batch_size: an int equal to the batch size.
        batches_per_epoch: an int equal to the number of batches
            in a single epoch.
    """

    # Using the same seed to shuffle both arrays maintains correspondence
    #   between the indices of images and their labels.
    seed = np.random.randint(0, 2**32)
    np.random.RandomState(seed).shuffle(images)
    np.random.RandomState(seed).shuffle(labels)
    batches = []
    for i in range(batches_per_epoch):
        if (i + 1) * batch_size < images.shape[0]:
            images_batch = images[(i * batch_size):((i + 1) * batch_size)]
            labels_batch = labels[(i * batch_size):((i + 1) * batch_size)]
        else: 
            images_batch = images[(i * batch_size):]
            labels_batch = labels[(i * batch_size):]
        batch = {'images': images_batch,
                 'labels': labels_batch}
        batches.append(batch)
    return batches


def auto_encode(image_data, label_data, model, num_samples, 
                fname=None):

    # visualize auto-encodings
    output_images = plt.figure(figsize=(5, 5), facecolor='#ffffff')
    for i in range(num_samples):
        proto_input = data[np.random.randint(0, data.shape[0])]
        input = np.empty((1, 794))
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
        noise = np.random.binomial(1, noise_parameter, size=(1, 794))
        first_ten = noise[0][0:10]
        print(first_ten)
        output = np.zeros((28, 28), dtype=np.float32)
        row_length = output.shape[1]
        for j in range(1, len(output.flatten())):
            noise = model.predict(noise, batch_size=1)
            p = noise[0][j+10 // row_length][j+10 % row_length]
            if p < 0:
                p = 0
            sample = np.random.binomial(1, p)
            noise[0][j+10 // row_length][j+10 % row_length] = p
            output[j // row_length][j % row_length] = sample
        subplot = generated_samples.add_subplot(plot_size, plot_size, i+1)
        subplot.imshow(output, cmap='gray')
        subplot.axis('off')
    if fname is None:
        plt.show()
    if fname is not None:
        plt.savefig(fname)


# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Binarize MNIST
x_test = np.digitize(x_test, [0, 128]) - 1
x_train = np.digitize(x_train, [0, 128]) - 1

# Do one-hot encoding of data labels
y_train = tf.one_hot(y_train, 10).numpy()
y_test = tf.one_hot(y_test, 10).numpy()

# Network parameters
batch_size = 100
num_epochs = 1
learning_rate = 0.001  # training parameter
epsilon = 0.000001  # training parameter
hidden_layers = 2
hidden_units = 2000
features = 784
num_masks = 1
classes = 10

batches_per_epoch = math.ceil(x_train.shape[0] / batch_size)
remainder_batch_size = x_train.shape[0] - ((batches_per_epoch - 1) * batch_size)
total_batches = batches_per_epoch * num_epochs

first_training = True  # set to True to train a new model
have_masks = False

# Generate or load masks
if have_masks:
    with open('semade_masks.txt', 'rb') as file:
        masks_dict = pkl.load(file)
    masks = masks_dict['masks']
    indices = masks_dict['indices']
    ss_indices = indices[0]
else:
    masks_dict = gen_masks(num_masks, features, classes, hidden_layers,
                           hidden_units)
    masks = masks_dict['masks']
    indices = masks_dict['indices']
    ss_indices = indices[0]

images = tf.keras.Input(shape=(28, 28), name='images')
labels = tf.keras.Input(shape=(10), name='labels')
images_flat = tf.keras.layers.Flatten()(images)
inputs = tf.keras.layers.Concatenate(axis=-1)([images_flat, labels])
h_1 = MaskedDense(hidden_units, masks[0][0], 'relu')(inputs)
dropout_1 = tf.keras.layers.Dropout(rate=0.15)(h_1)
h_2 = MaskedDense(hidden_units, masks[0][1], 'relu')(dropout_1)
dropout_2 = tf.keras.layers.Dropout(rate=0.15)(h_2)
h_out = MaskedDense(features, masks[0][2])(dropout_2)
direct_out = MaskedDense(features, masks[0][3])(inputs)
outputs = AddWithBias(features)([h_out, direct_out])
reshape = tf.keras.layers.Reshape(target_shape=(28, 28))(outputs)
SeMADE = Model(inputs=[images, labels], outputs=[reshape])
# tf.keras.utils.plot_model(SeMADE, 'semade_plot.png', show_shapes=True)
SeMADE.summary()
SeMADE.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
               loss=logx_loss)

if first_training is False:
    # ensure that network is properly initialized
    SeMADE.train_on_batch(x={'images': x_train[0:batch_size],
                             'labels': y_train[0:batch_size]},
                          y=x_train[0:batch_size])

    # load network weights from test file
    weights = np.load('semade_weights.npz', allow_pickle=True)['arr_0']
    SeMADE.set_weights(weights)

history = []
history_by_masks = []
for i in range(len(masks)):
    history_by_masks.append(i)
for i in range(num_epochs):
    epoch_data = batch_data(x_train, y_train, 100, batches_per_epoch)
    for j in range(len(epoch_data)):
        batch = epoch_data[j]
        # Train on one batch and add the loss to the history
        history.append(SeMADE.train_on_batch(x=batch, y=batch['images']))
        batches_completed = (batches_per_epoch * i) + j
    if np.std(history[-20:-1]) < 1:
        print('Loss: ' + str(history[-1]) + '. Training is terminating.')
        break
    else:
        print('Epoch ' + str(i + 1) + ' of ' + str(num_epochs) + '. Loss: ' +
              str(history[-1]))
        history_by_masks[batches_completed % len(masks)] = history[-1]
        next_mask_set = masks[(batches_completed + 1) % len(masks)]
        ss_indices = indices[(batches_completed + 1) % len(masks)]
        h_1_mask = next_mask_set[0]
        h_2_mask = next_mask_set[1]
        out_mask = next_mask_set[2]
        dir_mask = next_mask_set[3]
        h_1 = SeMADE.get_layer(index=4)
        h_2 = SeMADE.get_layer(index=6)
        out = SeMADE.get_layer(index=8)
        direct = SeMADE.get_layer(index=9)
        h_1.set_mask(h_1_mask)
        h_2.set_mask(h_2_mask)
        out.set_mask(out_mask)
        direct.set_mask(dir_mask)
      
# Plot training loss values
plt.plot(history)
plt.title('Model loss after ' + str(i + 1) + ' epochs')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()

# Save most recent losses for all masks
with open('semade_test_losses.txt', 'wb') as file:
        pkl.dump(history_by_masks, file)

# Save network after training
with open('semade_masks.txt', 'wb') as file:
    pkl.dump(masks, file)

weights = SeMADE.get_weights()
np.savez('semade_weights', weights)

auto_encode(test, SeMADE, 25, 'semade_ae')
# generate_samples(SeMADE, 25, 'semade_samples')
