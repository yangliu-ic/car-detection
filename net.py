import tensorflow as tf
import numpy as np


def make_unet(X, training):
    """
    Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor
    Returns:
        output (4-D Tensor): (N, H, W, C), same shape as the `input` tensor
    """
    with tf.variable_scope("layer0"):
        net = X / 127.5 - 1

    conv1, pool1 = conv_conv_pool(net, [8, 8], training, name=1)

    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)

    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)

    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)

    conv5 = conv_conv_pool(pool4, [128, 128], training, name=5, pool=False)

    up6 = deconv_concat(conv5, conv4, 128, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

    up7 = deconv_concat(conv6, conv3, 64, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

    up8 = deconv_concat(conv7, conv2, 32, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

    up9 = deconv_concat(conv8, conv1, 16, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, name=9, pool=False)

    return tf.layers.conv2d(conv9, 1, (1, 1), name='final',
                            activation=tf.nn.sigmoid, padding='same')

def conv_conv_pool(input_, n_filters, training, name, pool=True,
                   activation=tf.nn.relu):
    """
    {Conv -> BN -> RELU} x 2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_
    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None,
                                   padding='same', name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training,
                                                name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))
        if pool is False:
            return net
        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2),
                                       name="pool_{}".format(name))
        return net, pool

def deconv_concat(inputA, input_B, features, name):
    """
    Deconv `inputA` and concat with `input_B`
    """
    with tf.variable_scope("layer{}_deconv".format(name)):
        stddev = np.sqrt(2 / (3**2 * features))
        wd = weight_variable_devonc([2, 2, features//2, features], stddev)
        bd = bias_variable([features//2])
        upsample = tf.nn.relu(deconv2d(inputA, wd, 2, name) + bd,
                              name="relu{}_of_deconv".format(name))
        concat = tf.concat([upsample, input_B], axis=-1,
                           name="concat_{}".format(name))
    return concat

def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def deconv2d(x, W, stride, name):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2,
                            x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME', name="deconv")
