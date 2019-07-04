import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probabilityimport bijectors as tfb

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

made = AutoregressiveNetwork()