import operator
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph

import siuts

np.set_printoptions(threshold=np.inf)

num_classes = len(siuts.species_list)
dataset_loc = siuts.dataset_dir

siuts.create_dir("checkpoints/frozen_graphs/")


def accuracy(predictions, labels):
    return (1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def reformat(labels):
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return np.array(labels)


def load(fname):
    location = dataset_loc + fname + '.pickle'
    with open(location, 'rb') as f:
        images = pickle.load(f)
    print(fname + " loaded! " + str(images.shape))
    return images


testing_data = load("validation_data")
testing_labels = reformat(load("validation_labels"))
rec_ids = load("validation_rec_ids")

FLAGS = tf.app.flags.FLAGS
input_graph = "checkpoints/graph.pb"
input_saver = ""
# Whether the input files are in binary format.
input_binary = True

# The name of the output nodes, comma separated."
output_node_names = "sm_one,tf_one_prediction,sm_test,test_dataset_placeholder"

# The name of the master restore operator.
restore_op_name = "save/restore_all"

# The name of the tensor holding the save path
filename_tensor_name = "save/Const:0"

# Whether to remove device specifications.
clear_devices = True

# comma separated list of initializer nodes to run before freezing.
initializer_nodes = ""


def get_accuracies(graph_path, testing_labels, rec_ids):
    print ""
    print "Getting predictions..."

    accuracies = siuts.Accuracy()

    with tf.Session() as persisted_sess:
        with gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf_test_dataset = tf.placeholder(tf.float32, shape=(siuts.test_batch_size, 64, 64, 1))
            test_predictions_op = tf.import_graph_def(graph_def,
                                                      input_map={"test_dataset_placeholder:0": tf_test_dataset},
                                                      return_elements=['sm_test:0'])

        testing_predictions = np.empty
        for i in range(testing_data.shape[0] / siuts.test_batch_size):
            start = i * siuts.test_batch_size
            end = (i + 1) * siuts.test_batch_size
            if i == 0:
                testing_predictions = test_predictions_op[0].eval(feed_dict={tf_test_dataset: testing_data[start:end]})
            else:
                testing_predictions = np.concatenate((testing_predictions, test_predictions_op[0].eval(
                    feed_dict={tf_test_dataset: testing_data[start:end]})))

        testing_labels = testing_labels[:testing_predictions.shape[0]]

    predictions = np.argmax(testing_predictions, 1)
    labels = np.argmax(testing_labels, 1)
    rec_ids = rec_ids[:testing_predictions.shape[0]]

    print ""
    print "----Results for each segment----"

    accuracies.seg_acc = accuracy(testing_predictions, testing_labels)
    accuracies.seg_auc = roc_auc_score(testing_labels, testing_predictions, average="macro")
    accuracies.seg_f1 = f1_score(labels, predictions, average='macro')
    print "Accuracy: " + str(accuracies.seg_acc)
    print
    print "AUC score (weighted): " + str(roc_auc_score(testing_labels, testing_predictions, average="weighted"))
    print "AUC score (macro): " + str(accuracies.seg_auc)
    print "AUC score (micro): " + str(roc_auc_score(testing_labels, testing_predictions, average="micro"))

    print
    print "F1 score (weighted): " + str(f1_score(labels, predictions, average='weighted'))
    print "F1 score (macro): " + str(accuracies.seg_f1)
    print "F1 score (micro): " + str(f1_score(labels, predictions, average='micro'))

    print
    print
    accuracies.seg_conf_matrix = confusion_matrix(labels, predictions)

    print "Labels     Species name      Recall Precis. F1-score"
    print "----------------------------------------------------"
    for i in range(num_classes):
        TP = accuracies.seg_conf_matrix[i][i]
        SUM = np.sum(accuracies.seg_conf_matrix[i])
        if SUM == 0:
            recall = 0
            precision = 0
            f1 = 0
        else:
            recall = float(TP) / SUM * 100
            try:
                precision = float(TP) / np.sum(accuracies.seg_conf_matrix[:, i]) * 100
            except ZeroDivisionError:
                precision = 0

            try:
                f1 = 2 * (recall * precision) / (recall + precision)
            except ZeroDivisionError:
                f1 = 0

        print "{:2d} {:^25} {:05.2f} | {:05.2f} | {:05.2f}".format(i, siuts.species_list[i], recall, precision, f1)

    print
    print
    print

    file_predictions = []
    file_labels = []
    for rec_id in rec_ids:
        rec_predictions = []
        for i in range(len(rec_ids)):
            if rec_ids[i] == rec_id:
                rec_predictions.append(np.array(testing_predictions[i]))
                test_label = testing_labels[i]
        if (len(rec_predictions) > 0):
            file_predictions.append(np.array(rec_predictions))
            file_labels.append(test_label)

    file_predictions_mean = []
    for prediction in file_predictions:
        prediction = np.array(prediction)
        file_predictions_mean.append(np.asarray(np.mean(prediction, axis=0)))

    sum = 0
    for i in range(len(file_predictions_mean)):
        if (np.argmax(file_predictions_mean[i]) == np.argmax(file_labels[i])):
            sum += 1
    print "----Results for each recording----"

    accuracies.file_acc = float(sum) / len(file_predictions_mean)
    print "Accuracy: " + str(accuracies.file_acc)

    file_predictions_mean = np.array(file_predictions_mean)
    file_labels = np.array(file_labels)

    rec_predictions = np.array([np.argmax(x) for x in file_predictions_mean])
    rec_labels = np.argmax(file_labels, 1)

    accuracies.file_auc = roc_auc_score(file_labels, file_predictions_mean, average="macro")
    accuracies.file_f1 = f1_score(rec_labels, rec_predictions, average='macro')
    print
    print "AUC score (weighted): " + str(roc_auc_score(file_labels, file_predictions_mean, average="weighted"))
    print "AUC score (macro): " + str(accuracies.file_auc)
    print "AUC score (micro): " + str(roc_auc_score(file_labels, file_predictions_mean, average="micro"))

    print
    print "F1 score (weighted): " + str(f1_score(rec_labels, rec_predictions, average='weighted'))
    print "F1 score (macro): " + str(accuracies.file_f1)
    print "F1 score (micro): " + str(f1_score(rec_labels, rec_predictions, average='micro'))

    print
    print
    rec_conf_matrix = confusion_matrix(rec_labels, rec_predictions)
    accuracies.file_conf_matrix = rec_conf_matrix
    print
    print "Prediction accuracy by label"
    print "Labels     Species name       Recall Precis. F1-score"
    print "-----------------------------------------------------"
    for i in range(num_classes):
        TP = rec_conf_matrix[i][i]
        SUM = np.sum(rec_conf_matrix[i])
        try:
            recall = float(TP) / SUM * 100
        except ZeroDivisionError:
            recall = 0

        try:
            precision = float(TP) / np.sum(rec_conf_matrix[:, i]) * 100
        except ZeroDivisionError:
            precision = 0

        try:
            f1 = 2 * (recall * precision) / (recall + precision)
        except ZeroDivisionError:
            f1 = 0
        print "{:2d} {:^25} {:6.2f} | {:6.2f} | {:6.2f}".format(i, siuts.species_list[i], recall, precision, f1)

    print
    print
    print

    file_predictions_top = []
    for i in range(len(file_predictions_mean)):
        top_3 = []
        pred = np.copy(file_predictions_mean[i])
        for j in range(3):
            index = np.argmax(pred)
            top_3.append(index)
            pred[index] = -1.0
        file_predictions_top.append(top_3)

    TPs = 0
    FPs = 0
    for i in range(len(file_predictions_mean)):
        if rec_labels[i] in file_predictions_top[i]:
            TPs += 1
        else:
            FPs += 1
    accuracies.top3_acc = float(TPs) / len(file_predictions_mean)
    print "Top-3 accuracy: " + str(accuracies.top3_acc)
    return accuracies


output_path = "checkpoints/frozen_graphs/frozen_graph-{}.pb"
accuracies_list = []
for checkpoint in tf.train.get_checkpoint_state("checkpoints/").all_model_checkpoint_paths:
    step = checkpoint.split("-")[1]
    print
    print
    print
    print "###################################"
    print "   RESULTS FOR STEP {0}".format(step)
    print "###################################"
    freeze_graph.freeze_graph(input_graph, input_saver, input_binary, checkpoint, output_node_names,
                              restore_op_name, filename_tensor_name, output_path.format(step), clear_devices,
                              initializer_nodes)
    accuracies = get_accuracies(output_path.format(step), testing_labels, rec_ids)
    accuracies.step = step
    accuracies_list.append(accuracies)

with open("checkpoints/accuracies.pickle", 'wb') as f:
    pickle.dump(accuracies_list, f, protocol=-1)

print "    Metric                Value    Step"
print "-----------------------------------------"

accuracies_list.sort(key=operator.attrgetter('seg_acc'))
acc = accuracies_list[-1]
print "{:^25} {:1.4f} | {:6}".format("Segment accuracy", acc.seg_acc, acc.step)

accuracies_list.sort(key=operator.attrgetter('seg_auc'))
acc = accuracies_list[-1]
print "{:^25} {:1.4f} | {:6}".format("Segment AUC", acc.seg_auc, acc.step)

accuracies_list.sort(key=operator.attrgetter('seg_f1'))
acc = accuracies_list[-1]
print "{:^25} {:1.4f} | {:6}".format("Segment F1-score", acc.seg_f1, acc.step)

accuracies_list.sort(key=operator.attrgetter('file_acc'))
acc = accuracies_list[-1]
print "{:^25} {:1.4f} | {:6}".format("File accuracy", acc.file_acc, acc.step)

accuracies_list.sort(key=operator.attrgetter('file_auc'))
acc = accuracies_list[-1]
print "{:^25} {:1.4f} | {:6}".format("File AUC", acc.file_auc, acc.step)

accuracies_list.sort(key=operator.attrgetter('file_f1'))
acc = accuracies_list[-1]
print "{:^25} {:1.4f} | {:6}".format("File F1-score", acc.file_f1, acc.step)

accuracies_list.sort(key=operator.attrgetter('top3_acc'))
acc = accuracies_list[-1]
print "{:^25} {:1.4f} | {:6}".format("Top3 accuracy", acc.top3_acc, acc.step)

print
accuracies_list.reverse()
print "Mean of 5 highest top-3 accuracies: {}".format(np.mean([x.top3_acc for x in accuracies_list[:5]]))

accuracies_list.sort(key=operator.attrgetter('file_f1'))
accuracies_list.reverse()
print "Mean of 5 highest file F1 scores: {}".format(np.mean([x.file_f1 for x in accuracies_list[:5]]))
print
