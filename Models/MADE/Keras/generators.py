#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import math


def gen_masks(num_masks, features, hidden_layers, hidden_units):

    """Creates a desired number of sets of masks for MADE.

    Arguments:
        num_masks: an int equal to the number of different sets of
            masks desired.
        features: an int equal to the number of features of the data.
        hidden_layers: an int equal to the number of hidden layers in MADE.
        hidden_units: an int equal to the number of hidden units in MADE's
            hidden layers.

    Returns:
        A dictionary with keys 'masks' and 'indices'. Under key 'masks',
        there is a list containing num_masks lists of different masks for MADE.
        The masks in these lists are in the order: hidden layer 1, hidden layer
        2, output, and direct connection. Under key 'indices', there is a list
        containing the list of input/output node indices for each of the mask
        sets generated. The indices of the two lists in the dictionary
        correspond--the first mask set goes with the first set of indices,
        and so on.
    """

    # This array should contain numbers 1-784
    features_indices = []
    for i in range(features):
        features_indices.append(i + 1)
    masks = []
    indices = []
    for i in range(num_masks):
        set_masks = []  # Will contain all masks for the set
        # Randomize the input (and output, since they have to be the same)
        #   ordering
        set_features = []  # Input and output node indices for the set
        for index in features_indices:
            set_features.append(index)
        np.random.RandomState(np.random.randint(0, 2**32)).shuffle(
            set_features)
        indices.append(set_features)
        prev_indices = set_features
        for j in range(hidden_layers):
            layer_indices = []
            for k in range(hidden_units):
                # The hidden nodes' indices need to be between the minimum
                #   index from the previous layer and one less than the number
                #   of features, inclusive.
                layer_indices.append(np.random.randint(low=min(prev_indices),
                                                       high=features))
            mask = np.zeros((len(prev_indices), len(layer_indices)),
                            dtype=np.float32)
            for k in range(len(prev_indices)):
                for l in range(len(layer_indices)):
                    # The mask value will be one when the autoregressive
                    #   condition is met.
                    mask[k][l] = float(int(prev_indices[k] <= layer_indices[l]))
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            set_masks.append(mask)
            prev_indices = layer_indices
        output_mask = np.zeros((len(prev_indices), features), dtype=np.float32)
        for j in range(len(prev_indices)):
            for k in range(len(set_features)):
                output_mask[j][k] = float(int(prev_indices[j] < set_features[k]))
        output_mask = tf.convert_to_tensor(output_mask, dtype=tf.float32)
        set_masks.append(output_mask)
        direct_mask = np.zeros((features, features), dtype=np.float32)
        for j in range(features):
            for k in range(features):
                direct_mask[j][k] = float(int(set_features[j] < set_features[k]))
        direct_mask = tf.convert_to_tensor(direct_mask, dtype=tf.float32)
        set_masks.append(direct_mask)
        masks.append(set_masks)
    return{'masks': masks, 'indices': indices}


def auto_encode(model, data, num_samples, fname=None):

    # visualize auto-encodings
    output_images = plt.figure(figsize=(5, 5), facecolor='#ffffff')
    for i in range(num_samples):
        proto_input = data[np.random.randint(0, data.shape[0]+1)]
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


def compare_masks(model, num_examples, data, indices, masks,
                  masks_history, fname=None):
    """
    Generate autoencoded images and samples using the highest-performing
        variants of the model.

    Arguments:
        model: Keras implementation of a MADE.
        num_examples: an int representing the desired number of
            autoencoded images and samples.
        data: the dataset to be used for autoencoding.
        indices: a list of lists representing the orderings of the data
            features on which the model was trained.
        masks: a list of lists containing the masks with which the model
            was trained. Each of the items of this list corresponds to
            the item of indices with the same index.
        masks_history: a list containing the most recent loss values
            returned by the network using each of the masks in masks.
            Each item of this list corresponds to that of indices and
            masks with the same index.
        fname: the string containing the filename desired for any images
            saved. If a filename is provided, both autoencodings and samples
            will be saved. If no filename is provided, the two images will
            be displayed rather than saved.

    Returns:
        none.
    """

    # Ensure that data is a numpy array
    if tf.is_tensor(data):
        data.numpy()
    # Compute the number of different masks to be used
    num_masks = math.floor(math.sqrt(num_examples))
    plot_side = math.ceil(math.sqrt(num_examples))
    best_masks = []
    for i in range(num_masks):
        mask_index = masks_history.index(min(masks_history))
        ordering = indices[mask_index]
        mask_set = masks[mask_index]
        best_masks.append({'masks': mask_set,
                           'indices': ordering})
        del masks_history[mask_index]
    # Prepare noise for sample generation
    noise_shape = (plot_side,) + data.shape[1:]
    noise = np.random.rand(noise_shape[0], noise_shape[1],
                           noise_shape[2])
    # Prepare dataset for autoencoding
    selected_indices = np.random.randint(0, data.shape[0] + 1,
                                         plot_side).tolist()
    ae_data = []
    for index in selected_indices:
        ae_data.append(data[index])
    ae_data = np.asarray(ae_data)
    # Iterate through chosen masks
    h_1 = model.get_layer(index=2)
    h_2 = model.get_layer(index=4)
    out = model.get_layer(index=6)
    direct = model.get_layer(index=7)
    all_ae = []
    all_ss = []
    for mask in best_masks:
        indices = mask['indices']
        h_1_mask = mask['masks'][0]
        h_2_mask = mask['masks'][1]
        out_mask = mask['masks'][2]
        dir_mask = mask['masks'][3]
        h_1.set_mask(h_1_mask)
        h_2.set_mask(h_2_mask)
        out.set_mask(out_mask)
        direct.set_mask(dir_mask)
        # Generate samples with this mask
        samples = noise
        for i in range(1, len(indices) + 1):
            index = indices.index(i)
            row = index // data.shape[-1]
            col = index % data.shape[-1]
            x_out = model.predict(samples, batch_size=samples.shape[0])
            p = np.random.rand(samples.shape[0])
            for sample in range(samples.shape[0]):
                if x_out[sample][row][col] > p[sample]:
                    samples[sample][row][col] = 1.0
                else:
                    samples[sample][row][col] = 0.0
        # Autoencode with this mask
        autoencodings = model.predict(ae_data, batch_size=ae_data.shape[0])
        # Add each sample and autoencoding to its respective list
        for i in range(plot_side):
            all_ss.append(samples[i])
            all_ae.append(autoencodings[i])
    # Prepare images
    plot_ss = plt.figure(figsize=(10, 10), facecolor='#ffffff')
    for index in range(len(all_ss)):
        subplot_ss = plot_ss.add_subplot(plot_side, plot_side,
                                         index + 1)
        subplot_ss.imshow(all_ss[index], cmap='gray')
        subplot_ss.axis('off')
    if fname is None:
        plt.show(0)
    if fname is not None:
        ss_fname = fname + '_ss'
        plt.savefig(ss_fname)
    plot_ae = plt.figure(figsize=(10, 10), facecolor='#ffffff')
    for index in range(len(all_ae)):
        subplot_ae = plot_ae.add_subplot(plot_side, plot_side,
                                         index + 1)
        subplot_ae.imshow(all_ae[index], cmap='gray')
        subplot_ae.axis('off')
    if fname is None:
        plt.show()
    if fname is not None:
        ae_fname = fname + '_ae'
        plt.savefig(ae_fname)


def generate_samples(model, num_samples, indices, fname=None):
    samples = np.random.rand(num_samples, 28, 28)
    for i in range(1, len(indices) + 1):
        index = indices.index(i)
        row = index // 28
        col = index % 28
        x_out = model.predict(samples, batch_size=num_samples)
        p = np.random.rand(num_samples)
        for sample in range(num_samples):
            if x_out[sample][row][col] > p[sample]:
                samples[sample][row][col] = 1.0
            else:
                samples[sample][row][col] = 0.0
    plot_size = math.ceil(math.sqrt(num_samples))
    generated_samples = plt.figure(figsize=(10, 10), facecolor='#ffffff')
    for i in range(num_samples):
        subplot = generated_samples.add_subplot(plot_size, plot_size, i+1)
        subplot.imshow(samples[i], cmap='gray')
        subplot.axis('off')
    if fname is None:
        plt.show()
    if fname is not None:
        plt.savefig(fname)


def info_reorder(model, images, fname=None):

    rows = 1
    cols = len(images)
    if cols > 10:
        rows = math.ceil(cols / 10)
        cols = 10
    images_info = []
    result = plt.figure(figsize=(cols + 0.5, rows + (rows * 0.5)), facecolor='#000000')
    plt.title('Images by Decreasing Information Content',
              {'fontsize': 16, 'color': '#ffffff'})
    plt.axis('off')
    for i in images:
        input = np.empty(shape=(1, 28, 28))
        input[0] = i
        info = model.evaluate(x=input, y=input, batch_size=1, verbose=0)
        images_info.append(info)
    for i in range(len(images_info)):
        max_info = max(images_info)
        max_info_image = images[images_info.index(max_info)]
        image_plot = result.add_subplot(rows, cols, i + 1)
        image_plot.imshow(max_info_image, cmap='gray', )
        plt.axis('off')
        image_plot.set_title(str(round(max_info, 4)),
                             {'fontsize': 10}, color='#ffffff')
        images_info[images_info.index(max_info)] = -1
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, facecolor=result.get_facecolor())
    else:
        plt.show()


def ar_test(model, num_samples, indices):
    """
    This function prints the values of the pixel with index 1
    generated from num_samples different instances of noise.
    In a properly autoregressive network, these values will be
    equal, because the output node indicating the predicted
    probability of the first pixel's activation doesn't have
    any connections.
    """

    samples = np.random.rand(num_samples, 28, 28)
    i = 1
    index = indices.index(i)
    x_out = model.predict(samples, batch_size=num_samples)
    row = index // 28
    col = index % 28
    print(x_out[:, row, col])
