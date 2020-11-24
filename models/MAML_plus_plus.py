
import os
import random
import numpy as np
import sys
import tensorflow as tf

"""Convolutional layers used by MAML model."""
## compared to code from homework I made these changes:
## add maxpool
## remove global average pooling before fully connected layer. instead flatten

def conv_block(inp, cweight, bn, activation=tf.nn.relu, residual=False, training=True):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME')
    normed = bn(conv_output, training=False)
    normed = activation(normed)
    if training:
        unused = bn(conv_output, training=True) #update moving mean and variance
    return normed

class ConvLayers(tf.keras.layers.Layer):
    def __init__(self, channels, dim_hidden, dim_output, img_size, num_inner_updates=5):
        super(ConvLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size

        weights = {}

        dtype = tf.float32
        weight_initializer = tf.keras.initializers.GlorotUniform()
        k = 3

        weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1', dtype=dtype)
        weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2', dtype=dtype)
        weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3', dtype=dtype)
        weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4', dtype=dtype)
        weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden * 5 * 5, self.dim_output]), name='w5', dtype=dtype)
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        self.conv_weights = weights
        self.num_inner_updates = num_inner_updates
        self.bn_list_list = []
        for ii in range(num_inner_updates):
            bn_list = []
            for jj in range(4):
                bn_list.append(tf.keras.layers.BatchNormalization(name='bn' + str(ii) + "-" + str(jj)))
            self.bn_list_list.append(bn_list)

    def call(self, inp, weights, ii, training=True):
        channels = self.channels
        #print("convlayers inp shape", inp.shape)
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        #print("after reshape shape", inp.shape)
        hidden1 = conv_block(inp, weights['conv1'], self.bn_list_list[ii][0])
        hidden1 = tf.nn.max_pool(hidden1, ksize=2, strides=2, padding='VALID', name="maxpool1")
        hidden2 = conv_block(hidden1, weights['conv2'], self.bn_list_list[ii][1])
        hidden2 = tf.nn.max_pool(hidden2, ksize=2, strides=2, padding='VALID', name="maxpool2")
        hidden3 = conv_block(hidden2, weights['conv3'], self.bn_list_list[ii][2])
        hidden3 = tf.nn.max_pool(hidden3, ksize=2, strides=2, padding='VALID', name="maxpool3")
        hidden4 = conv_block(hidden3, weights['conv4'], self.bn_list_list[ii][3])
        hidden4 = tf.nn.max_pool(hidden4, ksize=2, strides=2, padding='VALID', name="maxpool4")
        # spatial dimensions are now 5x5 if starting from 84x84
        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        product = tf.matmul(hidden4, weights['w5'])
        with_bias = product + weights['b5']
        return with_bias

class MAMLpp(tf.keras.Model):
    def __init__(self, img_side_length, channels, dim_output,
                 inner_update_lr=0.4, num_filters=32, num_inner_updates=5):
        super(MAMLpp, self).__init__()
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.dim_hidden = num_filters
        self.channels = channels
        self.img_side_length = img_side_length
        self.num_inner_updates = num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        self.conv_layers = ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_side_length, num_inner_updates=num_inner_updates)

        self.inner_update_lr_dict = {}
        for key in self.conv_layers.conv_weights.keys():
            self.inner_update_lr_dict[key] = [tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for
                                              j in range(num_inner_updates)]

    def call(self, inp, max_parallel=25, multiply_by_five=False, training=True):

        @tf.function
        def task_inner_loop(inp):
            """
                Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
                Args:
                    inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
                      labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
                      labels used for evaluating the model after inner updates.
                Returns:
                  task_output: output for input_ts after training on input_tr
            """
            input_tr, input_ts, label_tr, label_ts = inp
            weights = self.conv_layers.conv_weights
            outputs_ts = []

            #for compatibility with an older version:

            if multiply_by_five:
                factor = 5.
            else:
                factor = 1.

            for ii in range(self.num_inner_updates):
                # print("innter loop input_tr shape", input_tr.shape)
                with tf.GradientTape(persistent=False) as inner_tape:
                    inner_tape.watch(weights)
                    outputs = self.conv_layers(input_tr, weights, ii, training=training)
                    loss_tr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=label_tr))
                grad = inner_tape.gradient(loss_tr, weights)
                weights = {w_name: w - factor * self.inner_update_lr_dict[w_name][ii] * grad[w_name] for w_name, w in weights.items()}
                outputs_ts.append(self.conv_layers(input_ts, weights, ii, training=training))
            return outputs_ts

        input_tr, input_ts, label_tr, label_ts = inp
        out_dtype_new = [tf.float32] * self.num_inner_updates
        result = tf.map_fn(task_inner_loop,
                           elems=(input_tr, input_ts, label_tr, label_ts),
                           dtype=out_dtype_new,
                           parallel_iterations=max_parallel)
        return result
