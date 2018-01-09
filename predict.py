import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

mycol = sns.husl_palette(10)[5]



def predict_from_trained_model(save_model_path, test_images):
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:

        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('input_features:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_preds = loaded_graph.get_tensor_by_name('accuracy/correct_prediction/prediction:0')

        predictions = [ ]
        softmax_probs = [ ]

        if np.ndim(test_images) == 4:
            for x in test_images:
                x = np.expand_dims(x, 0)
                test_feed = {loaded_x: x,
                             loaded_keep_prob: 1.0}
                softmax_prob = sess.run([ loaded_preds ], feed_dict=test_feed)
                predictions.append(np.argmax(softmax_prob))
                softmax_probs.append(softmax_prob)

        else:
            x = np.expand_dims(test_images, 0)
            test_feed = {loaded_x: x,
                         loaded_keep_prob: 1.0}
            softmax_prob = sess.run([ loaded_preds ], feed_dict=test_feed)
            predictions.append(np.argmax(softmax_prob))
            softmax_probs.append(softmax_prob)

    return predictions, softmax_probs


def show_predicted(save_model_path, test_image, true_label):
    _, softmax_probs = predict_from_trained_model(save_model_path, test_image)

    sign_names = pd.read_csv("traffic-signs-data/signnames.csv")
    softmax_probs = pd.DataFrame(softmax_probs[ 0 ][ 0 ].reshape(43, 1), columns=[ 'softmax_prob' ])

    tmp_table = pd.concat([ sign_names, softmax_probs ], 1)
    tmp_table = tmp_table.sort_values(by="softmax_prob", ascending=False)
    top5 = tmp_table.iloc[ :5, ]

    plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[ 4, 1 ])
    ax0 = plt.subplot(gs[ 0 ])
    ax0.imshow(np.squeeze(test_image), cmap="Greys")
    ax0.set_xticks(())
    ax0.set_yticks(())
    ax0.set_title("Preprocessed input image \n True label: {}".format(true_label), fontweight='bold')

    ax1 = plt.subplot(gs[ 1 ])
    sns.barplot(x="softmax_prob", y='SignName', data=top5, color=mycol, edgecolor=".2")
    ax1.set_xlim((0, 1))
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_yticks(())
    ax1.set_xticks(())

    ax1.set_title("Top 5 Predicted Traffic Signs", fontweight='bold')
    for i, p in enumerate(ax1.patches):
        ax1.annotate(top5.iloc[ i, 1 ],
                     (p.get_width() * 0.9, p.get_y() + .4), fontweight='bold')
        ax1.annotate(np.round(top5.iloc[ i, 2 ], 4), (p.get_width() * 0.9, p.get_y() + .6))
    plt.tight_layout()
    plt.show();

    return top5
