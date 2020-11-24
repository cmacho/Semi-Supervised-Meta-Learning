
import os
import random
import numpy as np
import sys
import tensorflow as tf
from functools import partial

"""Convolutional layers used by MAML model."""


class CustomBatchNorm(tf.keras.layers.Layer):
    """"an implementation of batch norm that behaves the same at test time as it does at train time, namely
     it uses batch statistics and not running statistics"""
    def __init__(self, dim):
        """ dim should be the number of channels i.e. the last dimension of
        expected input to the batchnorm layer of the form (batch, height, width, channels)"""
        super(CustomBatchNorm, self).__init__()
        dtype = tf.float32
        self.beta = tf.Variable(np.zeros((dim,)), dtype=dtype)
        self.gamma = tf.Variable(np.ones((dim,)), dtype=dtype)
        self.avg_mean = tf.constant(np.zeros((dim,)), dtype=dtype)
        self.avg_variance = tf.constant(np.ones((dim,)), dtype=dtype)
    def call(self, input, use_running_stats=False):
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keepdims=False)
        out = tf.nn.batch_normalization(input, mean, variance, self.beta, self.gamma, 0.00000001)
        return out

def conv_block(inp, cweight, bn, training=True, activation=tf.nn.relu, residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME')
    normed = bn(conv_output, training=True)
    return normed

class ConvLayers(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size, num_inner_updates=5):
        super(ConvLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size
        self.num_inner_updates = num_inner_updates

        weights = {}

        dtype = tf.float32
        weight_initializer = tf.keras.initializers.GlorotUniform()
        k = 3

        weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1', dtype=dtype)
        self.bn1 = CustomBatchNorm(self.dim_hidden)
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2', dtype=dtype)
        self.bn2 = CustomBatchNorm(self.dim_hidden)
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3', dtype=dtype)
        self.bn3 = CustomBatchNorm(self.dim_hidden)
        weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4', dtype=dtype)
        self.bn4 = CustomBatchNorm(self.dim_hidden)
        weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden * 5 * 5, self.dim_output]), name='w5', dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        self.conv_weights = weights
        self.num_inner_updates = num_inner_updates

    def call(self, inp, weights, training=True):
        channels = self.channels
        #print("convlayers inp shape", inp.shape)
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        #print("after reshape shape", inp.shape)
        hidden1 = conv_block(inp, weights['conv1'], self.bn1, training=training)
        hidden1 = tf.nn.max_pool(hidden1, ksize=2, strides=2, padding='VALID', name="maxpool1")
        hidden2 = conv_block(hidden1, weights['conv2'], self.bn2, training=training)
        hidden2 = tf.nn.max_pool(hidden2, ksize=2, strides=2, padding='VALID', name="maxpool2")
        hidden3 = conv_block(hidden2, weights['conv3'], self.bn3, training=training)
        hidden3 = tf.nn.max_pool(hidden3, ksize=2, strides=2, padding='VALID', name="maxpool3")
        hidden4 = conv_block(hidden3, weights['conv4'], self.bn4, training=training)
        hidden4 = tf.nn.max_pool(hidden4, ksize=2, strides=2, padding='VALID', name="maxpool4")
        # spatial dimensions are now 5x5 if starting from 84x84
        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        product = tf.matmul(hidden4, weights['w5'])
        with_bias = product + weights['b5']
        return with_bias

class MAML(tf.keras.Model):
    def __init__(self, img_side_length, channels, dim_output,
                 inner_update_lr=0.4, num_filters=32, num_inner_updates=5):
        super(MAML, self).__init__()
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.dim_hidden = num_filters
        self.channels = channels
        self.img_side_length = img_side_length
        self.num_inner_updates = num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        self.conv_layers = ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_side_length, num_inner_updates=num_inner_updates)


    def call(self, inp, num_inner_updates = -1, max_parallel=25, training=True):

        @tf.function
        def task_inner_loop(inp, num_inner_updates=num_inner_updates):
            input_tr, input_ts, label_tr, label_ts = inp
            weights = self.conv_layers.conv_weights
            outputs_ts = []

            for ii in range(num_inner_updates):
                # print("innter loop input_tr shape", input_tr.shape)
                with tf.GradientTape(persistent=False) as inner_tape:
                    inner_tape.watch(weights)
                    outputs = self.conv_layers(input_tr, weights, training=training)
                    loss_tr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=label_tr))
                grad = inner_tape.gradient(loss_tr, weights)
                weights = {w_name: w - self.inner_update_lr * grad[w_name] for w_name, w in weights.items()}
                outputs_ts.append(self.conv_layers(input_ts, weights, training=training))
            return outputs_ts

        input_tr, input_ts, label_tr, label_ts = inp

        if num_inner_updates == -1:
            num_inner_updates = self.num_inner_updates

        out_dtype_new = [tf.float32] * num_inner_updates

        inner_loop_partial = partial(task_inner_loop, num_inner_updates=num_inner_updates)

        result = tf.map_fn(inner_loop_partial,
                           elems=(input_tr, input_ts, label_tr, label_ts),
                           dtype=out_dtype_new,
                           parallel_iterations=max_parallel)
        return result
