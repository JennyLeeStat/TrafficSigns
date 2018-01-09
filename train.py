import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, confusion_matrix

import cnn_models as models



# model parameters =================

batch_size = 128
learning_rate = .001
epochs = 35
print_every = 100
evaluate_every = 500
checkpoint_every = 5
iteration = 0


# Checkpoint dir and log dir
checkpoint_dir = "checkpoint"
if not tf.gfile.Exists(checkpoint_dir):
    tf.gfile.MakeDirs(checkpoint_dir)

log_dir = "log"
if not tf.gfile.Exists(log_dir):
    tf.gfile.MakeDirs(log_dir)

save_model_path = 'checkpoint/cnn_small_kernel_deep_wide_s4w75_lr_001'

# ======================================================================


def train(X_train_large_gray, y_train_large_gray, X_valid_gray, y_valid_gray,
          iteration=iteration):

    def load_batch(X, y, batch_size=batch_size):
        for start in range(0, len(X), batch_size):
            end = min(len(X), start + batch_size)
            yield X[ start:end ], y[ start:end ]

    def evaluate(X_valid, y_valid):
        val_feed = {X_: X_valid,
                    y_: y_valid,
                    keep_prob_pl: 1.0,
                    is_training_: False}

        val_loss, val_acc = sess.run([ cost, accuracy ], feed_dict=val_feed)

        print("\n==================================="
              "\nEpoch: {}/{}".format(e + 1, epochs),
              " Iteration: {}".format(iteration),
              "\nValidation set loss: {:.4f}".format(val_loss),
              "\nValidation set accuracy: {:.4f}".format(val_acc),
              "\n===================================\n")

        return val_acc

    def test_accuracy(X_test, y_test):
        test_feed = {X_: X_test,
                     y_: y_test,
                     keep_prob_pl: 1.0,
                     is_training_: False}

        test_loss, test_acc = sess.run([ accuracy ], feed_dict=test_feed)

        print("\n==================================="
              "\nTest Acc: {:.4f}".format(test_acc),
              "\n===================================\n")


    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        # Inputs
        X_ = tf.placeholder(tf.float32, (None, 32, 32, 1), 'input_features')
        y_ = tf.placeholder(tf.int32, (None), 'labels')
        keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob")
        is_training_ = tf.placeholder(tf.bool, name="is_training")

        # CNN model
        logits = models.cnn_small_kernel_deep_wide(X_, keep_prob_pl, is_training_)
        logits = tf.identity(logits, name='logits')

        # Loss and optimizer
        one_hot_y = tf.one_hot(y_, depth=43)
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot_y, logits=logits)
            cost = tf.reduce_mean(cross_entropy, name='xentropy')

        with tf.name_scope('train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

                # Performance metrics: Accuracy
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                probability = tf.nn.softmax(logits, name='prediction')
                correct_pred = tf.equal(tf.argmax(probability, 1),
                                        tf.argmax(one_hot_y, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        tf.summary.scalar('xentropy', cost)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            train_writer = tf.summary.FileWriter('log', sess.graph)

            for e in range(epochs):
                #             for x, y in load_batch(X_train_color, y_train_color):
                for x, y in load_batch(X_train_large_gray, y_train_large_gray):
                    feed = {X_: x, y_: y, keep_prob_pl: 0.5, is_training_: True}
                    loss, train_accuracy, _, summary_ = sess.run([ cost, accuracy, optimizer, merged ], feed_dict=feed)
                    train_writer.add_summary(summary_, iteration)

                    if iteration % print_every == 0:
                        print("Epoch: {}/{}".format(e + 1, epochs),
                              "Iteration: {}".format(iteration),
                              "Training set loss: {:.4f}".format(loss),
                              "Training set accuracy: {:.4f}".format(train_accuracy))
                    iteration += 1
                # val_acc = evaluate(X_valid_color, y_valid_color)
                val_acc = evaluate(X_valid_gray, y_valid_gray)
                if e % checkpoint_every == 0 and e > 0:
                    save_model_path_ = save_model_path + "_epoch_" + str(e) + "_valacc_" + str(val_acc)
                    saver.save(sess, save_model_path_)
                    print("Model saved: {}".format(save_model_path_))



def test_model(save_model_path, test_features, test_labels):
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('input_features:0')
        loaded_y = loaded_graph.get_tensor_by_name('labels:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy/accuracy/accuracy:0')
        loaded_preds = loaded_graph.get_tensor_by_name('accuracy/correct_prediction/prediction:0')

        steps = 0
        stats = {
            'steps': [ ],
            'test_acc': [ ],
            'test_precision': [ ],
            'test_recall': [ ]
        }

        test_batch_acc_total = 0

        # Get accuracy in batches for memory limitations
        for x, y in models.load_batch(test_features, test_labels):
            steps += 1
            test_feed = {loaded_x: x,
                         loaded_y: y,
                         loaded_keep_prob: 1.0}

            test_preds, test_acc = sess.run([ loaded_preds, loaded_acc ], feed_dict=test_feed)
            test_precision = recall_score(y, np.argmax(test_preds, 1), average='weighted')
            test_recall = precision_score(y, np.argmax(test_preds, 1), average='weighted')

            stats[ 'steps' ].append(steps)
            stats[ 'test_acc' ].append(test_acc)
            stats[ 'test_precision' ].append(test_precision)
            stats[ 'test_recall' ].append(test_recall)

    return stats