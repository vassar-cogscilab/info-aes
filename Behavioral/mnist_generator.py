#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
         'eight', 'nine']

cwd = os.getcwd()
path = cwd + '/stimuli/'
os.chdir(path)
info = open('info.txt', 'w')

done = False
record = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # we want ten of each digit
while done is False:
    choice = np.random.randint(0, y_test.shape[0])  # pick from test data
    digit = y_test[choice]  # use pop to preclude duplicates
    data = x_test[choice]
    y_test = np.delete(y_test, choice, 0)  # mimic popping for np arrays
    x_test = np.delete(x_test, choice, 0)
    if record[digit] < 10:  # if we haven't yet saved ten of this digit
        fname = path + names[digit] + '_' + str(record[digit] + 1)
        img = plt.figure(figsize=(4, 4))
        ax = plt.Axes(img, [0., 0., 1., 1.])
        ax.set_axis_off()
        img.add_axes(ax)
        ax.imshow(data, cmap='gray', extent=(0, 4, 4, 0))
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        info.write(names[digit] + '_' + str(record[digit] + 1) +
            ': item ' + str(choice) + ' of test data\n')
        record[digit] += 1
    for i in range(len(record)):
        if min(record) == 10:
            done = True
info.close()