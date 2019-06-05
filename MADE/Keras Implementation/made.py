import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

# for now, we'll use MNIST as our dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# define binarize function for rank 1 tensors
def binarize(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                if x[i][j] > 0:
                    x[i][j] = 1
    return x

# binarize data (the MADE architecture is not equipped to process continuous
#   data)
for i in range(x_train.shape[0]): # iterate through every training pixel:
    x_train[i] = binarize(x_train[i])
    if i < x_test.shape[0]: # binarize the test data, too
        x_test[i] = binarize(x_test[i])

# do one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# implement masking as custom keras layer

# network parameters
batch_size = 256
hidden_units = 50

# make network
inputs = tf.keras.Input(shape=(28,28))
flatten = tf.keras.layers.Flatten()(inputs) # flatten matrix data to vectors
h_1 = tf.keras.layers.Dense(hidden_units, activation='relu')(flatten) #
h_2 = tf.keras.layers.Dense(hidden_units, activation='relu')(h_1)
h_3 = tf.keras.layers.Dense(hidden_units, activation='relu')(h_2)
outputs = tf.keras.layers.Dense(784, activation='sigmoid')(h_3)
unflatten = tf.keras.layers.Reshape((28,28))(outputs)
made = Model(inputs=inputs, outputs=unflatten)
made.summary()
made.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy'])

# train model
history=made.fit(x=x_train,y=x_train,batch_size=batch_size,epochs=20)

# visualize training
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# visualize auto-encoding capacity
choice = x_test[np.random.randint(0,x_test.shape[0]+1)]
input = np.empty((1,28,28))
input[0] = choice
output = made.predict(input,batch_size=1)
result = output[0]

plt.figure(figsize=(5,5))
plt.imshow(choice,cmap='gray')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(result,cmap='gray')
plt.show()
