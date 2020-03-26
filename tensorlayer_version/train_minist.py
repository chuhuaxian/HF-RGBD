import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import os
# initialize all viarable
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
x = tf.placeholder(tf.float32, shape=[64, 1], name='x___')
y_ = tf.placeholder(tf.int64, shape=[64], name='y___')

network = InputLayer(x, name='input')
network1 = DropoutLayer(network, keep=0.8, name='drop1')
network2 = DenseLayer(network1, 800, tf.nn.relu, name='Dense1')
network3 = DropoutLayer(network2, keep=0.5, name='drop2')
network4 = DenseLayer(network3, 800, tf.nn.relu, name='Dense2')
network5 = DropoutLayer(network4, keep=0.5, name='drop3')
network6 = DenseLayer(network5, 10, tf.identity, name='output')

# 定义损失函数
y = network6.outputs
y_op = tf.argmax(tf.nn.softmax(y), 1)
cost = tl.cost.cross_entropy(y, y_, name='entropy')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()

# 定义优化器
train_params = network6.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, var_list=train_params)

# 初始化模型参数
tl.layers.initialize_global_variables(sess)

# 训练参数设置
batch_size = 500
n_epoch = 100
print_freq = 5
tensor_board_train_index = 0
tensor_board_val_index = 0
import random
import time
tf.reset_default_graph()
for epoch in range(n_epoch):
    data_ = [random.randint(0, 50) for _ in range(64)]
    label_ = [ i% 10 for i in data_]
    data = np.expand_dims(np.array(data_), -1)
    data = data.astype(np.float32)
    label_ = np.array(label_)
    label_ = label_.astype(np.int64)
    start_time = time.time()
    print(data.dtype, label_.dtype)
    cost_, _ = sess.run([cost, train_op], {x: data, y_: label_})
    print(cost_)