import numpy as np
import tensorflow as tf
import scipy.misc


def batch_norm(x, scope):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=scope)

def conv2d(input, output_dim, f=4, stride=2, stddev=0.02, name="conv2d",pad='SAME'):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [f, f, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding=pad), bias)
        return conv

def deconv2d(input, output_shape, stride=2,k_h=4, k_w=4,  stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input, weight, output_shape=output_shape, strides=[1, stride, stride, 1]), bias)
        return deconv


def Identity_block_for_D(X, filters, stage='DIstg'):
    F1, F2, F3 = filters
    X_shortcut=X
    X1 = tf.nn.elu(batch_norm(conv2d(X,F1,f=1,stride=1,name=str(stage)+'A',pad='VALID'),str(stage)+'A'))
    X2 = tf.nn.elu(batch_norm(conv2d(X1,F2,f=4,stride=1,name=str(stage)+'B',pad='SAME'),str(stage)+'B'))
    X3 = batch_norm(conv2d(X2,F3,f=1,stride=1,name=str(stage)+'C',pad='VALID'),str(stage)+'C')
    X4 = tf.add(X_shortcut,X3)
    X5 = tf.nn.elu(X4)
    return X5

def Conv_block_for_D(X, filters ,s=2,stage='DCstg'):
    F1, F2, F3 = filters
    X_shortcut = X
    X1 = tf.nn.elu(batch_norm(conv2d(X,F1,f=4,stride=s,name=str(stage)+'A',pad='VALID'),str(stage)+'A'))
    X2 = tf.nn.elu(batch_norm(conv2d(X1,F2,f=1,stride=1,name=str(stage)+'B',pad='SAME'),str(stage)+'B'))
    X3 = batch_norm(conv2d(X2,F3,f=1,stride=1,name=str(stage)+'C',pad='VALID'),str(stage)+'C')
    X_shortcut_new = batch_norm(conv2d(X_shortcut,F3,f=1,stride=s,name=str(stage)+'D',pad='VALID'),str(stage)+'D')
    X4 = tf.add(X_shortcut_new,X3)
    X5 = tf.nn.elu(X4)
    return X5




def Identity_block_for_G(X, filters ,stage='Gstg'):
    F1, F2, F3 = filters
    X_shortcut = X
    X1 = tf.nn.elu(batch_norm(conv2d(X,F1,f=1,stride=1,name=str(stage)+'A',pad='VALID'),str(stage)+'A'))
    X2 = tf.nn.elu(batch_norm(conv2d(X1,F2,f=4,stride=1,name=str(stage)+'B',pad='SAME'),str(stage)+'B'))
    X3 = batch_norm(conv2d(X2,F3,f=1,stride=1,name=str(stage)+'C',pad='VALID'),str(stage)+'C')
    X4 = tf.add(X_shortcut,X3)
    X5 = tf.nn.elu(X4)
    return X5









