import tensorflow as tf
import pandas as pd
import numpy as np
import keras


def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


train_x = np.random.rand(100, 100)
train_y = np.random.randint(10, size=100)

n_hidden_1 = 38
n_input = train_x.shape[1]
n_classes = len(train_y)

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")

x1 = tf.constant([1,2,3])
x2 = tf.constant([1,2,3])
x3 = tf.placeholder(dtype=tf.float16)

y = tf.multiply(x1, x2)

sesh = tf.Session()

print(sesh.run(y))
