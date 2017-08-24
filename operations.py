import tensorflow as tf
import numpy as np
import os
import sys

def batch_wrapper(x, is_training, name='batch_norm', decay=0.99):
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [x.get_shape()[-1]], initializer=tf.constant_initializer(0))
        pop_mean = tf.get_variable('pop_mean', [x.get_shape()[-1]], initializer=tf.constant_initializer(0), trainable=False)
        pop_variance = tf.get_variable('pop_variance', [x.get_shape()[-1]], initializer=tf.constant_initializer(1), trainable=False)

        if is_training:
            batch_mean, batch_variance = tf.nn.moments(x, [0,1,2]) # Get moments from each channel axis
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1-decay))
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(x, batch_mean, batch_variance, offset=beta, scale=scale, variance_epsilon=1e-5)
        else:
            return tf.nn.batch_normalization(x, pop_mean, pop_variance, offset=beta, scale=scale, variance_epsilon=1e-5)

def conv2d(x, output_channel, filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='conv2d'):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter', [filter_height, filter_width, x.get_shape()[-1], output_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        convolution = tf.nn.conv2d(x, filter, strides=[1,stride_hor, stride_ver, 1], padding='SAME')
        biases = tf.get_variable('bias', [output_channel], initializer=tf.constant_initializer(0))
        weighted_sum = convolution + biases
        return weighted_sum

def deconv2d(x, output_shape, filter_height=4, filter_width=4, stride_hor=2, stride_ver=2, name='deconv2d'):
    with tf.variable_scope(name):
        filter = tf.get_variable('kernel', [filter_height, filter_width, output_shape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        deconvolution = tf.nn.conv2d_transpose(x, filter, output_shape=output_shape, strides=[1, stride_hor, stride_ver, 1])
        biases = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0))
        weighted_sum = deconvolution + biases
        return weighted_sum

def linear(x, h, name='linear'):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [x.get_shape()[-1], h], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [h], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(x, weight) + bias
        return weighted_sum

def leakyrelu(x, leak=0.1,name='leakyrelu'):
    return tf.maximum(leak*x, x)

def relu(x, name='relu'):
    return tf.maximum(x,0)

