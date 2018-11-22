import tensorflow as tf
import numpy as np
from PIL import Image



def conv(name, inputs, nums_out, k_size, strides=1, is_SN=False):
    nums_in = int(inputs.shape[-1])
    with tf.variable_scope(name):
        kernel = tf.get_variable("weights", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [nums_out], initializer=tf.constant_initializer(0.))
        if is_SN:
            inputs = tf.nn.conv2d(inputs, spectral_normalization(name, kernel), [1, strides, strides, 1], "SAME") + bias
        else:
            inputs = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "SAME") + bias
    return inputs

def conv_(inputs, w, b):
    return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def deconv(name, inputs, nums_out, k_size, strides=2):
    nums_in = int(inputs.shape[-1])
    B = tf.shape(inputs)[0]
    H = inputs.shape[1]
    W = inputs.shape[2]
    with tf.variable_scope(name):
        kernel = tf.get_variable("weights", [k_size, k_size, nums_out, nums_in], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [nums_out], initializer=tf.constant_initializer(0.))
        inputs = tf.nn.conv2d_transpose(inputs, kernel, [B, H * 2, W * 2, nums_out], [1, strides, strides, 1], "SAME") + bias
    return inputs

def B_residual_blocks(name, inputs, train_phase):
    temp = tf.identity(inputs)
    with tf.variable_scope(name):
        inputs = conv("conv1", inputs, 64, 3)
        inputs = batchnorm(inputs, train_phase, "BN1")
        inputs = prelu("alpha1", inputs)
        inputs = conv("conv2", inputs, 64, 3)
        inputs = batchnorm(inputs, train_phase, "BN2")
    return temp + inputs

def pixelshuffler(inputs, factor):
    B = tf.shape(inputs)[0]
    H = tf.shape(inputs)[1]
    W = tf.shape(inputs)[2]
    nums_in = int(inputs.shape[-1])
    nums_out = nums_in // factor ** 2
    inputs = tf.split(inputs, num_or_size_splits=nums_out, axis=-1)
    output = 0
    for idx, split in enumerate(inputs):
        temp = tf.reshape(split, [B, H, W, factor, factor])
        temp = tf.transpose(temp, perm=[0, 1, 4, 2, 3])
        temp = tf.reshape(temp, [B, H * factor, W * factor, 1])
        if idx == 0:
            output = temp
        else:
            output = tf.concat([output, temp], axis=-1)
    return output

def prelu(name, inputs):
    with tf.variable_scope(name):
        slope = tf.get_variable(name+"alpha", [1], initializer=tf.constant_initializer(0.01))
    return tf.maximum(inputs, inputs * slope)

def relu(inputs):
    return tf.nn.relu(inputs)

def tanh(inputs):
    return tf.nn.tanh(inputs)

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope * inputs, inputs)

def global_sum_pooling(inputs):
    return tf.reduce_sum(inputs, axis=[1, 2])

def batchnorm(x, train_phase, scope_bn):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def fully_connected(name, inputs, nums_out, is_SN=False):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name):
        W = tf.get_variable("weights", [inputs.shape[-1], nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("bias", [nums_out])
    if is_SN:
        return tf.matmul(inputs, spectral_normalization(name, W)) + b
    else:
        return tf.matmul(inputs, W) + b

def _l2normalize(v, eps=1e-12):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v)) + eps)


def max_singular_value(W, u=None, Ip=1):
    if u is None:
        u = tf.get_variable("u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False) #1 x ch
    _u = u
    _v = 0
    for _ in range(Ip):
        _v = _l2normalize(tf.matmul(_u, W), eps=1e-12)
        _u = _l2normalize(tf.matmul(_v, W, transpose_b=True), eps=1e-12)
    sigma = tf.reduce_sum(tf.matmul(_u, W) * _v)
    return sigma, _u, _v

def spectral_normalization(name, W, Ip=1):
    u = tf.get_variable(name + "_u", [1, W.shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)  # 1 x ch
    W_mat = tf.transpose(tf.reshape(W, [-1, W.shape[-1]]))
    sigma, _u, _ = max_singular_value(W_mat, u, Ip)
    with tf.control_dependencies([tf.assign(u, _u)]):
        W_sn = W / sigma
    return W_sn




