from ops import conv, relu, conv_, max_pooling, fully_connected, batchnorm, prelu, leaky_relu, B_residual_blocks, pixelshuffler
import tensorflow as tf
import numpy as np

class generator:
    def __init__(self, name, B):
        self.name = name
        self.B = B

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            inputs = conv("conv1", inputs, 64, 9)
            inputs = prelu("alpha1", inputs)
            skip_connection = tf.identity(inputs)
            #The paper has 16 residual blocks
            for b in range(1, self.B + 1):
                inputs = B_residual_blocks("B"+str(b), inputs, train_phase)
            # inputs = B_residual_blocks("B2", inputs, train_phase)
            # inputs = B_residual_blocks("B3", inputs, train_phase)
            # inputs = B_residual_blocks("B4", inputs, train_phase)
            # inputs = B_residual_blocks("B5", inputs, train_phase)
            inputs = conv("conv2", inputs, 64, 3)
            inputs = batchnorm(inputs, train_phase, "BN")
            inputs = inputs + skip_connection
            inputs = conv("conv3", inputs, 256, 3)
            inputs = pixelshuffler(inputs, 2)
            inputs = prelu("alpha2", inputs)
            inputs = conv("conv4", inputs, 256, 3)
            inputs = pixelshuffler(inputs, 2)
            inputs = prelu("alpha3", inputs)
            inputs = conv("conv5", inputs, 3, 9)
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # inputs = tf.random_crop(inputs, [-1, 70, 70, 3])
            inputs = conv("conv1_1", inputs, 64, 3, 2)
            inputs = leaky_relu(inputs, 0.2)
            # inputs = conv("conv1_2", inputs, 64, 3, is_SN=True)
            # inputs = leaky_relu(inputs, 0.2)
            inputs = conv("conv2_1", inputs, 128, 3, 2)
            inputs = batchnorm(inputs, train_phase, "BN1")
            inputs = leaky_relu(inputs, 0.2)
            # inputs = conv("conv2_2", inputs, 128, 3, is_SN=True)
            # inputs = leaky_relu(inputs, 0.2)
            inputs = conv("conv3_1", inputs, 256, 3, 2)
            inputs = batchnorm(inputs, train_phase, "BN2")
            inputs = leaky_relu(inputs, 0.2)
            # inputs = conv("conv3_2", inputs, 256, 3, is_SN=True)
            # inputs = leaky_relu(inputs, 0.2)
            inputs = conv("conv4_1", inputs, 512, 3, 2)
            inputs = batchnorm(inputs, train_phase, "BN3")
            inputs = leaky_relu(inputs, 0.2)
            # inputs = fully_connected("fc", inputs, 512, is_SN=True)
            output = fully_connected("output", inputs, 1)
        return output

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

def vggnet(inputs, vgg_path):
    inputs = (inputs + 1) * 127.5
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])
    para = np.load(vgg_path+"vgg19.npy", encoding="latin1").item()
    inputs = relu(conv_(inputs, para["conv1_1"][0], para["conv1_1"][1]))
    inputs = relu(conv_(inputs, para["conv1_2"][0], para["conv1_2"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv_(inputs, para["conv2_1"][0], para["conv2_1"][1]))
    inputs = relu(conv_(inputs, para["conv2_2"][0], para["conv2_2"][1]))
    F = inputs
    inputs = max_pooling(inputs)
    inputs = relu(conv_(inputs, para["conv3_1"][0], para["conv3_1"][1]))
    inputs = relu(conv_(inputs, para["conv3_2"][0], para["conv3_2"][1]))
    inputs = relu(conv_(inputs, para["conv3_3"][0], para["conv3_3"][1]))
    inputs = max_pooling(inputs)
    inputs = relu(conv_(inputs, para["conv4_1"][0], para["conv4_1"][1]))
    inputs = relu(conv_(inputs, para["conv4_2"][0], para["conv4_2"][1]))
    inputs = relu(conv_(inputs, para["conv4_3"][0], para["conv4_3"][1]))
    return F


