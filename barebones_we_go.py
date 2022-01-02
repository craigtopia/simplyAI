print('iunderstand')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# import sklearn
# May 1st, 2020

# solve ax + b = y

# tensor flow is:
# declare constants, variables, placeholders
# draw computation graph
# initialize
# sess.run


# create placeholders
# initialize params
# forward prop
# cost
# backprop
# tf initialize variables

m = 10000
nx = (1, m)
ny = (1, m)
nh = 15
nL = 5

alpha = 0.0001
n_epochs = 30000
eps = 1e-6

x_feed = np.random.rand(nx[0], nx[1])

# Choose a Funky Function Here!
# y_feed = np.log(x_feed + eps)
y_feed = x_feed**2 + 7
# y_feed = (y_feed - np.mean(y_feed)) / np.std(y_feed)

# create placeholders
x = tf.placeholder(dtype=tf.float32, name='x', shape=(nx[0], None))
y = tf.placeholder(dtype=tf.float32, name='y', shape=(ny[0], None))

# initialize params
w1 = tf.get_variable(name='w1', shape=[nh, nx[0]], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name='b1', shape=[nh, 1], initializer=tf.zeros_initializer())
w2 = tf.get_variable(name='w2', shape=[nh, nh], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable(name='b2', shape=[nh, 1], initializer=tf.zeros_initializer())
w3 = tf.get_variable(name='w3', shape=[nL, nh], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable(name='b3', shape=[nL, 1], initializer=tf.zeros_initializer())
w4 = tf.get_variable(name='w4', shape=[1, nL], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable(name='b4', shape=[1, 1], initializer=tf.zeros_initializer())
parameters = {'w1': w1,
              'b1': b1,
              'w2': w2,
              'b2': b2,
              'w3': w3,
              'b3': b3,
              'w4': w4,
              'b4': b4}

# forward prop
z1 = tf.matmul(w1, x) + b1
# a1 = tf.nn.tanh(z1)
# custom relu activation function
masked = tf.greater(z1, 0)
zeros = tf.zeros_like(z1)
a1 = tf.where(masked, z1, zeros)  # relu

z2 = tf.matmul(w2, a1) + b2
a2 = tf.nn.tanh(z2)

z3 = tf.matmul(w3, a2) + b3
a3 = tf.keras.activations.relu(z3)

z4 = tf.matmul(w4, a3) + b4
a4 = tf.keras.activations.linear(z4)

# compute cost
cost = tf.linalg.norm(a4 - y)

# back prop
optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

# predictor
tf.estimator.Estimator.predict()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    costs = []
    for i in range(n_epochs):
        _, opt_cost = sess.run([optimizer, cost], feed_dict={x: x_feed,
                                                             y: y_feed})
        if i % 100 == 0:
            print(i, opt_cost)
            costs.append(opt_cost)

    # save params
    opt_params = sess.run(parameters)

    est, yo, xo = sess.run([a4, y, x], feed_dict={x: x_feed,
                                                  y: y_feed})

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(costs)
    axes[1].plot(xo[0], yo[0], linestyle='', marker='.', color='blue')
    axes[1].plot(xo[0], est[0], linestyle='', marker='.', color='red', alpha=0.5)
    plt.show()

