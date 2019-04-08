#! usr/bin/env python3

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import _pickle as cPickle
import gzip
np.set_printoptions(threshold=np.nan)
import time

#define parameters
batch_size = 128
features = 784
hidden_layers = 2
hidden_units = 500

random_init = False

weights = np.load('4_1made_weightsv10.npz')


def node_index(nodes, features, h, prev_nodes=[1]):
  if h == 0 or h == hidden_layers+1:
    indexes = np.arange(1,features+1)
    np.random.shuffle(indexes)
  else:
    indexes = np.random.randint(np.amin(prev_nodes),high=((features+1)-h),size=nodes)
  return indexes

def in_mask(prev_nodes,prev_h,indexes):
  indexes=indexes
  mask = np.zeros((len(prev_nodes),len(indexes)))
  for i in range(len(prev_nodes+1)):
    for j in range(len(indexes+1)):
      if indexes[j] >= prev_nodes[i]:
        mask[i,j] = 1
  return mask

def out_mask(prev_nodes,prev_h,indexes):
  indexes = indexes
  mask = np.zeros((len(prev_nodes),len(indexes)))
  for i in range(len(prev_nodes+1)):
    for j in range(len(indexes+1)):
      if indexes[j] > prev_nodes[i]:
        mask[i,j] = 1
  return mask
"""
masks = dict()
for i in range(2):
    in_indexes = node_index(features,features,0)
    h1_indexes = node_index(hidden_units,features,0+1,prev_nodes=in_indexes)
    h2_indexes = node_index(hidden_units,features,1+1,prev_nodes=h1_indexes)
    out_indexes = in_indexes
    h1_mask = in_mask(in_indexes,0,h1_indexes)
    h2_mask = in_mask(h1_indexes,1,h2_indexes)
    out_m = out_mask(h2_indexes,2,out_indexes)
    dir_m = out_mask(in_indexes, 0, out_indexes)
    masks[str(i)] = [h1_mask, h2_mask, out_m, dir_m]

# save masks
#np.savez("4_1made_masks", m = masks)
f = open("4_1made_masks.pkl", "wb")
cPickle.dump(masks,f)
f.close()
"""
#f = open("4_1made_masks.pkl", "rb")
test = cPickle.load(open("4_1made_masks.pkl", "rb"))
print(type(test['0'][0]))
print(test['0'][0].shape)

tf.reset_default_graph()

x = tf.placeholder(tf.float32,shape=(batch_size,features)) #images

if random_init:
  #h1 weight and bias
  w1 = tf.get_variable("w1",shape=(features,hidden_units),initializer=tf.random_normal_initializer(0,0.5))
  b1 = tf.get_variable("b1",shape=(1,hidden_units),initializer=tf.random_normal_initializer(0,0.5))

  #h2 weight and bias
  w2 = tf.get_variable("w2",shape=(hidden_units,hidden_units),initializer=tf.random_normal_initializer(0,0.5))
  b2 = tf.get_variable("b2",shape=(1,hidden_units),initializer=tf.random_normal_initializer(0,0.5))

  #output layer weight and bias
  x_hat = tf.get_variable("x_hat",shape=(hidden_units,features),initializer=tf.random_normal_initializer(0,0.0000005))
  x_b_hat = tf.get_variable("x_b_hat",shape=(1,features),initializer=tf.random_normal_initializer(0,0.0000005))

  #direct connection
  dirr = tf.get_variable("dirr",shape=(features,features),initializer=tf.random_normal_initializer(0,0.0000005))
else:
  #h1 weight and bias
  w1 = tf.get_variable("w1",shape=(features,hidden_units),initializer=tf.constant_initializer(weights['w1']))
  b1 = tf.get_variable("b1",shape=(1,hidden_units),initializer=tf.constant_initializer(weights['b1']))

  #h2 weight and bias
  w2 = tf.get_variable("w2",shape=(hidden_units,hidden_units),initializer=tf.constant_initializer(weights['w2']))
  b2 = tf.get_variable("b2",shape=(1,hidden_units),initializer=tf.constant_initializer(weights['b2']))

  #output layer weight and bias
  x_hat = tf.get_variable("x_hat",shape=(hidden_units,features),initializer=tf.constant_initializer(weights['x_hat']))
  x_b_hat = tf.get_variable("x_b_hat",shape=(1,features),initializer=tf.constant_initializer(weights['x_b_hat']))

  #direct connection
  dirr = tf.get_variable("dirr",shape=(features,features),initializer=tf.constant_initializer(weights['dirr']))

#create network
hidden1 = tf.nn.relu(tf.add(b1,tf.matmul(x,tf.multiply(w1,h1_mask))))
hidden2 = tf.nn.relu(tf.add(b2,tf.matmul(hidden1,tf.multiply(w2,h2_mask))))
out = tf.nn.sigmoid(tf.add(tf.add(x_b_hat,tf.matmul(hidden2,tf.multiply(x_hat,out_m))), tf.matmul(x,tf.multiply(dirr,dir_m))))

def cross_entropy(x, y, axis=-1):
  safe_y = tf.where(tf.equal(x, 0.), tf.ones_like(y), y)
  return -tf.reduce_sum(x * tf.log(safe_y), axis)

def entropy(x, axis=-1):
  return cross_entropy(x, x, axis)

#loss function: binary cross entropy
loss = entropy(out)
####loss = tf.reduce_sum(-x*tf.log(out)-(1-x)*tf.log(1-out))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
init = tf.global_variables_initializer()



"""
def gen_image(num_images):
	x = np.random.rand(num_images, features)
	
	for i in range(0,features):
		hidden1 = tf.nn.relu(tf.add(weights['b1'],tf.matmul(x,tf.multiply(weights['w1'],masks[
"""



