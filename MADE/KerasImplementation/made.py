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

# for now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# binarize
x_test = np.digitize(x_test, [0, 32]) - 1
x_train = np.digitize(x_train, [0, 32]) - 1

# mnist_samples = plt.figure(figsize=(5,5), facecolor='#ffffff')
# for i in range(64):
#     image = x_train[np.random.randint(0,x_train.shape[0]+1)]
#     subplot = mnist_samples.add_subplot(math.sqrt(64), math.sqrt(64), i+1)
#     subplot.imshow(image,cmap='gray')
#     subplot.axis('off')
# plt.show()

# network parameters
batch_size = 100
num_epochs = 1
learning_rate = 0.01 # training parameter
epsilon = 0.000001 # training parameter
num_masks = 64
hidden_layers = 2
hidden_units = 150
features = 784

batches_per_epoch = math.ceil(x_train.shape[0] / batch_size)
remainder_batch_size = x_train.shape[0] - ((batches_per_epoch-1) * batch_size)
total_batches = batches_per_epoch * num_epochs


# generate an array of mask matrices
def gen_masks(num_masks, features, hidden_layers, hidden_units):


    all_masks = []
    first_input_indices = []
    for j in range(features):
        first_input_indices.append(j+1)
    input_indices = []
    input_dim = hidden_units
    for m in range(num_masks):
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
        for j in range(len(input_indices)): # iterate over every layer node
            for k in range(len(first_input_indices)): # iterate over every input node
                if input_indices[j] < first_input_indices[k]:
                    output_mask[j][k] = 1
        output_mask = tf.convert_to_tensor(output_mask, dtype=tf.float32)
        masks.append(output_mask)
        direct_mask = np.zeros((features, features), dtype=np.float32)
        for j in range(len(first_input_indices)): # iterate over every layer node
            for k in range(len(first_input_indices)): # iterate over every input node
                if first_input_indices[j] <= first_input_indices[k]:
                    direct_mask[j][k] = 1
        direct_mask = tf.convert_to_tensor(direct_mask, dtype=tf.float32)
        masks.append(direct_mask)
        all_masks.append(masks)
    return all_masks

# visualize auto-encoding and generative capacities
# auto-encoding
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
    

# novel sample generation
def zoom_generation(start, end, step):


    plots = math.ceil(((end - start) / step) + 1)
    plot_size = math.ceil(math.sqrt(plots))
    noise = np.random.binomial(1,0.5,size=(1,28,28))
    row_length = noise.shape[1]
    input = noise[0]
    history_image = plt.figure(figsize=(10,10), facecolor='#ffffff')
    subplot = history_image.add_subplot(plot_size, plot_size, 1)
    subplot.imshow(input,cmap='gray')
    subplot.axis('off')
    for i in range(start, end + 1): 
        output = made.predict(noise,batch_size=1)
        p = output[0][i//row_length][i%row_length]
        sample = np.random.binomial(1, p)
        output[0][i//row_length][i%row_length] = sample
        if i % step == 0:
            plot_number = ((end - i )// step) + 1
            subplot = history_image.add_subplot(plot_size, plot_size, plot_number)
            subplot.imshow(output[0],cmap='gray')
            subplot.axis('off')
    plt.show()


def generate_sample_with_history(noise_parameter):


    noise = np.random.binomial(1,noise_parameter,size=(1,28,28))
    row_length = noise.shape[1]
    input = noise[0]
    plt.figure(figsize=(5,5))
    plt.imshow(input,cmap='gray')
    plt.show()

    output = np.zeros(input.shape, dtype=np.float32)
    history_image = plt.figure(figsize=(8,8), facecolor='#ffffff')
    for i in range(1, len(input.flatten())): 
        noise = made.predict(noise,batch_size=1)
        p = noise[0][i//row_length][i%row_length]
        sample = np.random.binomial(1, p)
        output[i//row_length][i%row_length] = sample
        if i % 16 == 0:
            plot_number = i // 16 + 1
            subplot = history_image.add_subplot(7, 7, plot_number)
            subplot.imshow(output,cmap='gray')
            subplot.axis('off')
    plt.show()

    #visualize output
    plt.figure(figsize=(5,5))
    plt.imshow(output,cmap='gray')
    plt.show()


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

# generate set of masks
masks = gen_masks(num_masks, features, hidden_layers, hidden_units)

# make network
inputs = tf.keras.Input(shape=(28,28))
flatten = tf.keras.layers.Flatten()(inputs) # flatten matrix data to vectors
h_1 = MaskedDense(hidden_units, masks[0][0], 'relu')(flatten) 
h_2 = MaskedDense(hidden_units, masks[0][1], 'relu')(h_1)
h_out = MaskedDense(784, masks[0][2])(h_2)
direct_out = MaskedDense(784, masks[0][3])(flatten)
merge = tf.keras.layers.Add()([h_out, direct_out])
outputs = tf.keras.layers.Activation('sigmoid')(merge)
unflatten = tf.keras.layers.Reshape((28,28))(outputs)
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
        np.random.shuffle(x_train)
        if len(data) * batch_size + batch_size > x_train.shape[0]:
            batch = x_train[len(data) * batch_size:]
        else:
            batch = x_train[len(data) * batch_size:(len(data) * batch_size) + batch_size]
        data.append(batch)
    print('Epoch ' + str(i) + ' of ' + str(num_epochs) + '.')
    for j in range(batches_per_epoch):
        if j % 100 == 0:
            print('Batch ' + str(j) + ' of ' + str(batches_per_epoch) + '.')
        batches_completed = (batches_per_epoch * i) + j 
        next_mask_set = batches_completed % len(masks) + 1
        # train on one batch and add the loss to the history
        history.append(made.train_on_batch(x=data[j],y=data[j]))
        # take weights from MADE
        weights = made.get_weights()
        tf.keras.backend.clear_session()
        inputs = tf.keras.Input(shape=(28,28))
        # remake network with next mask set
        inputs = tf.keras.Input(shape=(28,28))
        flatten = tf.keras.layers.Flatten()(inputs) # flatten matrix data to vectors
        h_1 = MaskedDense(hidden_units, masks[next_mask_set][0], 'relu')(flatten) 
        h_2 = MaskedDense(hidden_units, masks[next_mask_set][1], 'relu')(h_1)
        h_out = MaskedDense(784, masks[next_mask_set][2])(h_2)
        direct_out = MaskedDense(784, masks[next_mask_set][3])(flatten)
        merge = tf.keras.layers.Add()([h_out, direct_out])
        outputs = tf.keras.layers.Activation('sigmoid')(merge)
        unflatten = tf.keras.layers.Reshape((28,28))(outputs)
        made = Model(inputs=inputs, outputs=unflatten)
        # restore previous training
        made.set_weights(weights)
        made.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate, epsilon),
            loss='binary_crossentropy')  
# Plot training loss values
plt.plot(history)
plt.title('Model ' + str(i) +' loss')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()  