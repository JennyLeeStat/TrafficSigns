import tensorflow as tf


def conv2d_maxpool(features, filters, layer_name,
                   conv_kernel_size=(5, 5), conv_strides=1,
                   pool_kernel_size=(2, 2), pool_strides=2,
                   padding='same', is_training=True):
    with tf.name_scope(layer_name):
        net = tf.layers.conv2d(
            features,
            filters=filters,
            kernel_size=conv_kernel_size,
            strides=conv_strides,
            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
            padding='same',
            activation=tf.nn.relu)
        tf.summary.histogram('activations', net)

        # net = tf.layers.batch_normalization(net, training=is_training)

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
    net = conv2d_maxpool(features, 6, 'Conv1', padding='valid')
    net = conv2d_maxpool(net, 16, 'Conv2', padding='valid')
    net = flatten(net)
    net = tf.layers.dense(net, 120, name='FC1', activation=tf.nn.relu)
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    logits = tf.layers.dense(net, 43, name='Out')
    return logits

def cnn_base(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16, 'Conv1')
    net = conv2d_maxpool(net, 32, 'Conv2')
    net = tf.nn.dropout(net, keep_prob, name='dropout')
    net = flatten(net)
    net = tf.layers.dense(net, 256, name='FC1', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout3')
    logits = tf.layers.dense(net, 43, name='Out')
    return logits

def cnn_deep(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16, 'Conv1')
    net = conv2d_maxpool(net, 32, 'Conv2')
    net = conv2d_maxpool(net, 64, 'Conv3')
    net = tf.nn.dropout(net, keep_prob, name='dropout')
    net = flatten(net)
    net = tf.layers.dense(net, 256, name='FC1', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout3')
    logits = tf.layers.dense(net, 43, name='Out')
    return logits

def cnn_wide(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 32, 'Conv1')
    net = conv2d_maxpool(net, 64, 'Conv2')
    net = tf.nn.dropout(net, keep_prob, name='dropout')
    net = flatten(net)
    net = tf.layers.dense(net, 256, name='FC1', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout3')
    logits = tf.layers.dense(net, 43, name='Out')
    return logits

def cnn_wide2(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16, 'Conv1')
    net = conv2d_maxpool(net, 32, 'Conv2')
    net = tf.nn.dropout(net, keep_prob, name='dropout')
    net = flatten(net)
    net = tf.layers.dense(net, 1024, name='FC1', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    logits = tf.layers.dense(net, 43, name='Out')
    return logits

def cnn_small_kernel(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16, 'Conv1', conv_kernel_size=(3, 3))
    net = conv2d_maxpool(net, 32, 'Conv2', conv_kernel_size=(3, 3))
    net = tf.nn.dropout(net, keep_prob, name='dropout')
    net = flatten(net)
    net = tf.layers.dense(net, 256, name='FC1', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout3')
    logits = tf.layers.dense(net, 43, name='Out')
    return logits

def cnn_small_kernel_less_dropout(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16, 'Conv1', conv_kernel_size=(3, 3))
    net = conv2d_maxpool(net, 32, 'Conv2', conv_kernel_size=(3, 3))
    net = flatten(net)
    net = tf.layers.dense(net, 256, name='FC1', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout2')
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout3')
    logits = tf.layers.dense(net, 43, name='Out')
    return logits

def cnn_wide_deep(features, keep_prob=.5, is_training=True):
    net = conv2d_maxpool(features, 16, 'Conv1')
    net = conv2d_maxpool(features, 32, 'Conv2')
    net = conv2d_maxpool(net, 64, 'Conv3')
    net = flatten(net)
    net = tf.layers.dense(net, 256, name='FC1', activation=tf.nn.relu)
    net = tf.layers.dense(net, 84, name='FC2', activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob, name='dropout3')
    logits = tf.layers.dense(net, 43, name='Out')
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



def cnn_model(features, keep_prob=0.5, is_training=True):
    initializer = tf.truncated_normal_initializer(0.0, 0.1)
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

    # Convolutional Layer #1
    net = tf.layers.conv2d(
        features,
        filters=16,
        kernel_size=[ 5, 5 ],
        strides=1,
        padding='same',
        kernel_initializer=initializer,
        activation=tf.nn.relu)
    # net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.max_pooling2d(net, pool_size=[ 2, 2 ], strides=2)

    # Convolutional Layer #2
    net = tf.layers.conv2d(
        net,
        filters=32,
        kernel_size=[ 5, 5 ],
        strides=1,
        padding='same',
        kernel_initializer=initializer,
        activation=tf.nn.relu)
    # net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.max_pooling2d(net, pool_size=[ 2, 2 ], strides=2)

    # Convolutional Layer #3
    net = tf.layers.conv2d(
        net,
        filters=64,
        kernel_size=[ 3, 3 ],
        strides=1,
        padding='same',
        kernel_initializer=initializer,
        activation=tf.nn.relu)
    # net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.layers.max_pooling2d(net, pool_size=[ 2, 2 ], strides=2)
    net = tf.nn.dropout(net, keep_prob)

    # Fully connected layer
    shape = net.get_shape().as_list()[ 1: ]
    flat_dim = shape[ 0 ] * shape[ 1 ] * shape[ 2 ]
    net = tf.reshape(net, [ -1, flat_dim ])
    net = tf.layers.dense(net, units=256, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)

    # Fully connected layer
    net = tf.layers.dense(net, units=84, activation=tf.nn.relu)
    net = tf.nn.dropout(net, keep_prob)

    # Logits Layer
    logits = tf.layers.dense(net, units=43)

    return logits
