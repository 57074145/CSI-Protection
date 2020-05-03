# -*- coding: utf-8 -*-
"""
CSI location CNN_softmax classifer
@author: Wei Zhang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import h5py  
import numpy as np
import gc

from sklearn import preprocessing

"""
Load data

"""
data_path = '../dataset/three_model_AAEN.mat'

data = h5py.File(data_path,'r')
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train_x = np.transpose(data['train_x'][:])
test_x = np.transpose(data['test_x'][:])

num_classes = 6


def norm(train_x, test_x):
    
    train_x = train_x.reshape(-1, 20*90)
    train_x_max = np.max(train_x, axis=0)
    train_x_min = np.min(train_x, axis=0)
    norm_train_x = (train_x-train_x_min)/(train_x_max-train_x_min)
    test_x = test_x.reshape(-1, 20*90) 

    norm_test_x = (test_x-train_x_min)/(train_x_max-train_x_min)
    
    return norm_train_x, norm_test_x

train_x, test_x = norm(train_x,test_x)   #Nomlalization

train_x = train_x.reshape(-1,20, 90,1)
test_x = test_x.reshape(-1, 20, 90,1)
train_y = np.transpose(data['train_loc'][:])
test_y = np.transpose(data['test_loc'][:])

#del data
gc.collect()

"""
Define functions

"""


def TrainandTest(faces,label):
    array=np.arange(label.shape[0])
    test_num=round(0.25*label.shape[0])
    np.random.shuffle(array)
    faces=faces[array]
    label=label[array]
    
    train_x=faces[test_num:]
    train_y=label[test_num:]
    test_x=faces[0:test_num]
    test_y=label[0:test_num]
    
    return train_x,train_y,test_x,test_y

batch_size=128
def generate_testbatch(X,Y, batch_size):
    for batch_i in range(Y.shape[0]//batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys # generate test batch

        
def get_batch(faces,label,batch_size):
    array=np.arange(label.shape[0])
    np.random.shuffle(array)
    idx=array[:batch_size]
    batch_xs = faces[idx]
    batch_ys = label[idx]
    return batch_xs, batch_ys # generate batch


def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)*0.1 
    return tf.get_variable(name = name, initializer = initial)

def bias_variable(name, shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.get_variable(name = name, initializer = initial)

def conv2d_SAME(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_VALID(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def avg_pool_2x2(x):

    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

"""
CNN_softmax 

"""                          
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 20, 90, 1])

"""
# Convolutional Layer
# Relu activation function
# Pooling layer

"""
W_conv1 = weight_variable("W_conv1", [3, 9, 1, 16])  
b_conv1 = bias_variable("b_conv1", [16])                    
h_conv1 = tf.nn.relu(conv2d_VALID(x, W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable("W_conv2", [2, 8, 16, 32])
b_conv2 = bias_variable("b_conv2", [32])
h_conv2 = tf.nn.relu(conv2d_VALID(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable("W_conv3", [3, 6, 32, 64])
b_conv3 = bias_variable("b_conv3", [64])
h_conv3 = tf.nn.relu(conv2d_VALID(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


"""
Fully connected layer

"""

W_fc1 = weight_variable("W_fc1", [ 1*6*64, 128])
b_fc1 = bias_variable("b_fc1", [128])
h_pool2_flat = tf.reshape(h_pool3, [-1, 1*6*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32) 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""
Softmax layer

""" 
W_fc2 = weight_variable("W_fc2", [128, num_classes])
b_fc2 = bias_variable("b_fc2", [num_classes])

y_ = tf.placeholder(tf.float32, [None, num_classes])


y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # softmax classifiers

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,tf.reduce_max(y_conv))), reduction_indices=[1]))# softmax loss

loss = cross_entropy

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)




correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer()) # Variable initialization


saver=tf.train.Saver()

checkpoin_dir = "location/"

ckpt = tf.train.get_checkpoint_state(checkpoin_dir)
if ckpt and ckpt.model_checkpoint_path:
    print("checkpoint_path: " + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")

"""
 Run CNN

"""
max_iter= 10000
eoch_loss=np.zeros((max_iter,1))


"""
 Train CNN

"""
for i in range(0,max_iter):
    batch_x,batch_y = get_batch(train_x, train_y,batch_size)
    
    feed_dict = {x: batch_x, y_: batch_y, keep_prob: 1.0}
    
    _, eoch_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
    
    if (i)%100 == 0:
     
        feed_dict = {x: batch_x, y_: batch_y, keep_prob: 1.0}
        
        train_accuracy = sess.run(accuracy,feed_dict=feed_dict)
        print("step : %d, training accuracy : %g"%(i, train_accuracy))
        print("step : %d, loss : %g"%(i, eoch_loss))
        print("--------------------------------------------------")
        
    if (i+1) % 100 == 0: 
        saver.save(sess, "location/cnn_net.ckpt", i)     
      
res=[]
index = 1;
sum_test_accuracy = 0;
test_batch_size = 300;

"""
 Test CNN

"""
for batch_xs,batch_ys in generate_testbatch(test_x, test_y,test_batch_size):
    
    feed_dict = {x: batch_xs, y_: batch_ys, keep_prob:  1.0}
     
    temp_accuracy = sess.run(accuracy, feed_dict=feed_dict)
    print("index : %d, temp accuracy : %g"%( index, temp_accuracy))
    
feed_dict = {x: test_x, y_: test_y, keep_prob:  1.0} 
test_accuracy = sess.run(accuracy, feed_dict=feed_dict) 
print("test accuracy : %g"%( test_accuracy))



