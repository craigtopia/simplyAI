print('hi')
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

a = 3.
b = 5.

m = 10000
nx = (1, m)
ny = (1, m)
nh = 1

alpha = 0.01
n_epochs = 10000

x_feed = np.random.rand(nx[0], nx[1])
y_feed = a*x_feed + b

# create placeholders
x = tf.placeholder(dtype=tf.float32, name='x', shape=(nx[0], None))
y = tf.placeholder(dtype=tf.float32, name='y', shape=(ny[0], None))

# initialize params
w1 = tf.get_variable(name='w1', shape=[nh, nx[0]], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable(name='b1', shape=[nh, 1], initializer=tf.zeros_initializer())
parameters = {'w1': w1,
              'b1': b1}

# forward prop
z1 = tf.matmul(w1, x) + b1

# compute cost
cost = tf.linalg.norm(z1 - y)

# back prop \
optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    costs = []
    for i in range(n_epochs):
        _, opt_cost = sess.run([optimizer, cost], feed_dict={x: x_feed, y: y_feed})
        if i % 100 == 0:
            print(i, opt_cost)
            costs.append(opt_cost)

    optimal_weights = sess.run([w1, b1])

    print(optimal_weights)
    plt.plot(costs)
    plt.show()
