#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
# from MADE.KerasImplementation.layers import MaskedDense
# from layers import MaskedDense
from KerasImplementation.layers import MaskedDense
import numpy as np
import matplotlib.pyplot as plt
import math


def auto_encode(model, num_samples, fname=None):


    # visualize inputs
    original_images = plt.figure(figsize=(5,5), facecolor='#ffffff')
    # visualize auto-encodings
    output_images = plt.figure(figsize=(5,5), facecolor='#ffffff')
    for i in range(num_samples):
        proto_input = x_test[np.random.randint(0,x_test.shape[0]+1)]
        subplot = original_images.add_subplot(math.sqrt(num_samples), math.sqrt(num_samples), i+1)
        subplot.imshow(proto_input,cmap='gray')
        subplot.axis('off')
        input = np.empty((1,28,28))
        input[0] = proto_input
        output = model.predict(input,batch_size=1)
        subplot = output_images.add_subplot(math.sqrt(num_samples), math.sqrt(num_samples), i+1)
        subplot.imshow(output[0],cmap='gray')
        subplot.axis('off')
    if fname == None:
        plt.show()
    if fname != None:
        plt.savefig(fname)


def generate_samples(model, num_samples, fname=None):


    noise_parameter = np.random.rand()
    plot_size = math.ceil(math.sqrt(num_samples))
    generated_samples = plt.figure(figsize=(10,10), facecolor='#ffffff')
    for i in range(num_samples):
        noise = np.random.binomial(1,noise_parameter,size=(1,28,28))
        output = np.zeros(noise[0].shape, dtype=np.float32)
        row_length = noise.shape[1]
        for j in range(1, len(noise.flatten())): 
            noise = model.predict(noise,batch_size=1)
            p = noise[0][j//row_length][j%row_length]
            sample = np.random.binomial(1, p)
            noise[0][j//row_length][j%row_length] = p
            output[j//row_length][j%row_length] = sample
        subplot = generated_samples.add_subplot(plot_size, plot_size, i+1)
        subplot.imshow(output,cmap='gray')
        subplot.axis('off')
    if fname == None:
        plt.show()
    if fname != None:
        plt.savefig(fname)

# for now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

inputs = tf.keras.Input(shape=(28,28))
flatten = tf.keras.layers.Flatten()(inputs)
made_zero = tf.keras.models.load_model(
    'KerasImplementation/Tuning/6_11_made_0.h5', 
    custom_objects={'MaskedDense': MaskedDense})
made_one = tf.keras.models.load_model(
    'KerasImplementation/Tuning/6_11_made_1.h5', 
    custom_objects={'MaskedDense': MaskedDense})
made_two = tf.keras.models.load_model(
    'KerasImplementation/Tuning/6_11_made_2.h5', 
    custom_objects={'MaskedDense': MaskedDense})
made_three = tf.keras.models.load_model(
    'KerasImplementation/Tuning/6_11_made_3.h5', 
    custom_objects={'MaskedDense': MaskedDense})
merge = keras.layers.Average([made_zero, made_one, made_two, made_three])
multi_made = Model(inputs=inputs, outputs=merge)

multi_made.summary()