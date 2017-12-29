import tensorflow as tf


def cnn_model(features, keep_prob=0.5, is_training=True):
    initializer = tf.truncated_normal_initializer(0.0, 0.1)

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


import tensorflow as tf


def cnn_model_gray(features, keep_prob=0.5, is_training=True):
    initializer = tf.truncated_normal_initializer(0.0, 0.1)

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
