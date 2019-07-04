#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import math

# generate an array of mask matrices
def gen_masks(num_masks, features, hidden_layers, hidden_units, classes=None):

    if classes is not None:
        label_indices = []
        for i in range(classes):
            label_indices.append(0)
    first_input_indices = []
    for i in range(features):
        first_input_indices.append(i+1)
    all_masks = []
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
                layer_indices.append(np.random.randint(min(input_indices), features - 1))
            mask = np.zeros((len(input_indices), len(layer_indices)), dtype=np.float32)
            # populate mask with appropriate values
            for j in range(len(input_indices)):  # iterate over every layer node
                for k in range(len(layer_indices)):  # iterate over every input node
                    if input_indices[j] <= layer_indices[k]:
                        mask[j][k] = 1
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            masks.append(mask)
            input_indices = layer_indices
        # generate output layer masks
        output_mask = np.zeros((len(input_indices), features), dtype=np.float32)
        for j in range(len(input_indices)):  # every layer node
            for k in range(len(first_input_indices)):  # every input node
                if input_indices[j] < first_input_indices[k]:
                    output_mask[j][k] = 1
        output_mask = tf.convert_to_tensor(output_mask, dtype=tf.float32)
        masks.append(output_mask)
        direct_mask = np.zeros((len(first_input_indices), len(first_input_indices)), 
                               dtype=np.float32)
        for j in range(len(first_input_indices)):
            for k in range(len(first_input_indices)):
                if first_input_indices[j] < first_input_indices[k]:
                    direct_mask[j][k] = 1
        direct_mask = tf.convert_to_tensor(direct_mask, dtype=tf.float32)
        masks.append(direct_mask)
        all_masks.append(masks)
    return all_masks



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
                             {'fontsize': 10}, color = '#ffffff')
        images_info[images_info.index(max_info)] = -1
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, facecolor=result.get_facecolor())
    else:
        plt.show()