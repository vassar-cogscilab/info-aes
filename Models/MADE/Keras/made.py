#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras.datasets import mnist
from layers import MaskedDense
from layers import AddWithBias
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle as pkl
import generators
from tensorflow.keras.utils import multi_gpu_model

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Binarize MNIST
x_test = np.digitize(x_test, [0, 128]) - 1
x_train = np.digitize(x_train, [0, 128]) - 1


def batch_data(data, batch_size, batches_per_epoch):
    """ 
    Produces a list of batches for one epoch from a training dataset.

    Arguments:
        data: an array or tensor containing all the training data.
        batch_size: an int equal to the batch size.
        batches_per_epoch: an int equal to the number of batches
            in a single epoch.
    """

    batches = []
    np.random.shuffle(data)
    for i in range(batches_per_epoch):
        if (i + 1) * batch_size < data.shape[0]:
            batch = data[(i * batch_size):((i + 1) * batch_size)]
        else:
            batch = data[(i * batch_size):]
        batches.append(batch)
    return batches

# Hyperparameters
data = x_train
batch_size = 100
num_epochs = 0
batches_per_epoch = math.ceil(data.shape[0] / batch_size)
remainder_batch_size = data.shape[0] - ((batches_per_epoch - 1) * batch_size)
total_batches = batches_per_epoch * num_epochs
learning_rate = 0.01
epsilon = 0.00001
hidden_layers = 2
hidden_units = 8000
features = 784
num_masks = 64
first_training = False
have_masks = True

# Generate or load masks
if have_masks:
    with open('made_final.txt', 'rb') as file:
        masks_dict = pkl.load(file)
        masks = masks_dict['masks']
        indices = masks_dict['indices']
        ss_indices = indices[0]

else:
    masks_dict = generators.gen_masks(num_masks, features, hidden_layers,
                                      hidden_units)
    masks = masks_dict['masks']
    indices = masks_dict['indices']
    ss_indices = indices[0]
    with open('made_final.txt', 'wb') as file:
        pkl.dump(masks_dict, file)


def logx_loss(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    xent_loss = 28 * 28 * metrics.binary_crossentropy(y_true_flat, y_pred_flat)
    return xent_loss

inputs = tf.keras.Input(shape=(28, 28))
flatten = tf.keras.layers.Flatten()(inputs)
h_1 = MaskedDense(hidden_units, masks[0][0], 'relu')(flatten)
dropout_1 = tf.keras.layers.Dropout(rate=0.15)(h_1)
h_2 = MaskedDense(hidden_units, masks[0][1], 'relu')(dropout_1)
dropout_2 = tf.keras.layers.Dropout(rate=0.15)(h_2)
h_out = MaskedDense(features, masks[0][2], use_bias=False)(dropout_2)
direct_out = MaskedDense(features, masks[0][3], use_bias=False)(flatten)
outputs = AddWithBias(features)([h_out, direct_out])
unflatten = tf.keras.layers.Reshape(target_shape=(28, 28))(outputs)
made = Model(inputs=inputs, outputs=unflatten)
made.summary()
# made = multi_gpu_model(made, gpus=8)
made.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
             loss=logx_loss)

history = []
history_by_masks = []

if first_training is False:
    # Ensure that network is properly initialized
    made.train_on_batch(x=data[0:batch_size], y=data[0:batch_size])

    # Load network weights from test file
    weights = np.load('made_final.npz', allow_pickle=True)['arr_0']
    made.set_weights(weights)

    # Load mask history from file
    with open('made_final_losses.txt', 'rb') as file:
        history_by_masks = pkl.load(file)

else:
    for i in range(len(masks)):
        history_by_masks.append(i)

for i in range(num_epochs):
    epoch_data = batch_data(data, 100, batches_per_epoch)
    for j in range(len(epoch_data)):
        batch = epoch_data[j]
        # Train on one batch and add the loss to the history
        history.append(made.train_on_batch(x=batch, y=batch))
        batches_completed = (batches_per_epoch * i) + j + 1
    if np.std(history[-20:-1]) < 1:
        print('Loss: ' + str(history[-1]) + '. Training is terminating.')
        break
    else:
        history_by_masks[batches_completed % len(masks)] = history[-1]
        next_mask_set = masks[(batches_completed + 1) % len(masks)]
        ss_indices = indices[(batches_completed + 1) % len(masks)]
        h_1_mask = next_mask_set[0]
        h_2_mask = next_mask_set[1]
        out_mask = next_mask_set[2]
        dir_mask = next_mask_set[3]
        h_1 = made.get_layer(index=2)
        h_2 = made.get_layer(index=4)
        out = made.get_layer(index=6)
        direct = made.get_layer(index=7)
        h_1.set_mask(h_1_mask)
        h_2.set_mask(h_2_mask)
        out.set_mask(out_mask)
        direct.set_mask(dir_mask)
    print('Epoch ' + str(i + 1) + ' of ' + str(num_epochs) +
          ' complete. Loss: ' + str(history[-1]))

# Plot training loss values
plt.plot(history)
plt.title('Model loss after ' + str(i + 1) + ' epochs')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()

# Save most recent losses for all masks
with open('made_final_losses.txt', 'wb') as file:
    pkl.dump(history_by_masks, file)

# Save trained network
weights = made.get_weights()
np.savez('made_final', weights)

generators.compare_masks(model=made, num_examples=100, data=x_test,
                         indices=indices, masks=masks,
                         masks_history=history_by_masks, fname='made_final')
# print(ss_indices[0:10])  # Verify that the model's used random orderings
# generators.ar_test(model=made, num_samples=100, indices=ss_indices)
# generators.auto_encode(model=made, data=data, num_samples=100,
#                        fname='made_final_ae')
# print("Auto encoding complete.")
# generators.generate_samples(made, 100, ss_indices, 'made_final_ss')
# print("All sampling complete.")
