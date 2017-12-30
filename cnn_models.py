import tensorflow as tf
def conv2d_maxpool(features, filters,
                   conv_kernel_size=(5, 5), conv_strides=1,
                   pool_kernel_size=(2, 2), pool_strides=2,
                   padding='same', is_training=True):

    net = tf.layers.conv2d(
        features,
        filters=filters,
        kernel_size=conv_kernel_size,
        strides=conv_strides,
        kernel_initializer= tf.truncated_normal_initializer(0.0, 0.1),
        padding='same',
        activation=tf.nn.relu)
    #net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.max_pooling2d(
        net,
        pool_size=pool_kernel_size,
        strides=pool_strides,
        padding='same'
    )
    return net


def flatten(features):
    shape = features.get_shape().as_list()[1:]
    flat_dim = shape[0] * shape[1] * shape[2]
    return tf.reshape(features, [-1, flat_dim])


def lenet(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 6, padding='valid')
    net = conv2d_maxpool(net, 16, padding='valid')
    net = flatten(net)
    net = tf.layers.dense(net, 120, activation=tf.nn.relu)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 43)
    return logits


def cnn_base(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16)
    net = conv2d_maxpool(net, 32)
    net = tf.nn.dropout(net, keep_prob)
    net = flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    logits = tf.layers.dense(net, 43)
    return logits


def cnn_deep(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16)
    net = conv2d_maxpool(net, 32)
    net = conv2d_maxpool(net, 64)
    net = tf.nn.dropout(net, keep_prob)
    net = flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    logits = tf.layers.dense(net, 43)
    return logits


def cnn_wide(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 32)
    net = conv2d_maxpool(net, 64)
    net = tf.nn.dropout(net, keep_prob)
    net = flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    logits = tf.layers.dense(net, 43)
    return logits


def cnn_wide2(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16)
    net = conv2d_maxpool(net, 32)
    net = tf.nn.dropout(net, keep_prob)
    net = flatten(net)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    logits = tf.layers.dense(net, 43)
    return logits


def cnn_small_kernel(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16, conv_kernel_size=(3, 3))
    net = conv2d_maxpool(net, 32, conv_kernel_size=(3, 3))
    net = tf.nn.dropout(net, keep_prob)
    net = flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    logits = tf.layers.dense(net, 43)
    return logits


def cnn_best(features, keep_prob=0.5, is_training=True):
    net = conv2d_maxpool(features, 16)
    net = conv2d_maxpool(net, 32)
    net = conv2d_maxpool(net, 64, conv_kernel_size=(3, 3))
    net = tf.nn.dropout(net, keep_prob)
    net = flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    net = tf.layers.dense(net, 84, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)
    logits = tf.layers.dense(net, 43)
    return logits