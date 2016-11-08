
# coding: utf-8

# # Evaluation results
# Balanced dataset by oversampling by multiplying the datasets and then copying random samples until excactly balanced. Each class has mean of all all samples segments

# In[ ]:

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle
from tensorflow.python.platform import gfile
from sklearn.metrics import confusion_matrix
import numpy as np
np.set_printoptions(threshold=np.inf)

num_classes = 20
test_batch = 100
dataset_loc = "../data/dataset/1/testing/"

with open('../data/labels.pickle', 'rb') as f:
    labels_dict = pickle.load(f)

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

testing_data = load("testing_data")
testing_labels = load("testing_labels")
rec_ids = load("testing_rec_ids")

#testing_data = load("testing_data_2")
#testing_labels = load("testing_labels_2")
#rec_ids = load("testing_rec_ids_2")

print "Nr of files: " + str(np.max(rec_ids))

# In[ ]:

with tf.Session() as persisted_sess:
    with gfile.FastGFile("logs/frozen_graph.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf_test_dataset = tf.placeholder(tf.float32, shape=(test_batch, 64, 64, 1))
        test_predictions_op = tf.import_graph_def(graph_def, 
                                                input_map={"test_dataset_placeholder:0": tf_test_dataset},
                                                return_elements = ['sm_test:0'])


    testing_predictions = np.empty
    for i in range(testing_data.shape[0]/test_batch):
        start = i*test_batch
        end = (i+1)*test_batch
        if i == 0:
            testing_predictions = test_predictions_op[0].eval(feed_dict={tf_test_dataset: testing_data[start:end]})
        else:
            testing_predictions = np.concatenate((testing_predictions, test_predictions_op[0].eval(feed_dict={tf_test_dataset: testing_data[start:end]})))

        print "Testing predictions: " + str(i) + " - " + str(testing_predictions.shape)

    testing_labels = testing_labels[:testing_predictions.shape[0]]

#print testing_labels
predictions = np.argmax(testing_predictions, 1)
labels = np.argmax(testing_labels, 1)
rec_ids = rec_ids[:testing_predictions.shape[0]]
print np.max(labels)
print len(np.unique(labels))



print "Accuracy: " + str(accuracy(testing_predictions, testing_labels))
print
print "AUC score (weighted): " + str(roc_auc_score(testing_labels, testing_predictions, average="weighted"))
print "AUC score (macro): " + str(roc_auc_score(testing_labels, testing_predictions, average="macro"))
print "AUC score (micro): " + str(roc_auc_score(testing_labels, testing_predictions, average="micro"))

print
print "F1 score (weighted): " + str(f1_score(labels, predictions, average='weighted'))
print "F1 score (macro): " + str(f1_score(labels, predictions, average='macro'))
print "F1 score (micro): " + str(f1_score(labels, predictions, average='micro'))

print
print
conf_matrix = confusion_matrix(labels, predictions)
print "Confusion Matrix"
print conf_matrix
print
print "Prediction accuracy by label"

print "Labels     Species name      Recall Precis. F1-score"
print "----------------------------------------------------"
for i in range(num_classes):
    TP = conf_matrix[i][i]
    SUM = np.sum(conf_matrix[i])
    if SUM == 0:
        recall = 0
        precision = 0
        f1 = 0
    else:
        recall = float(TP)/SUM*100
        precision = float(TP)/np.sum(conf_matrix[:,i])*100
        f1 = 2*(recall*precision)/(recall+precision)
    precision = float(TP)/np.sum(conf_matrix[:,i])*100
    print "{:2d} {:^25} {:05.2f} | {:05.2f} | {:05.2f}".format(i, labels_dict[i], recall, precision, f1)


print
print
print




file_predictions = []
file_labels = []
for rec_id in range(max(rec_ids) + 1):
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
print "Accuracy: " + str(float(sum)/len(file_predictions_mean))


file_predictions_mean = np.array(file_predictions_mean)
file_labels = np.array(file_labels)

rec_predictions = np.array([np.argmax(x) for x in file_predictions_mean])
rec_labels = np.argmax(file_labels, 1)

print
print "AUC score (weighted): " + str(roc_auc_score(file_labels, file_predictions_mean, average="weighted"))
print "AUC score (macro): " + str(roc_auc_score(file_labels, file_predictions_mean, average="macro"))
print "AUC score (micro): " + str(roc_auc_score(file_labels, file_predictions_mean, average="micro"))

print
print "F1 score (weighted): " + str(f1_score(rec_labels, rec_predictions, average='weighted'))
print "F1 score (macro): " + str(f1_score(rec_labels, rec_predictions, average='macro'))
print "F1 score (micro): " + str(f1_score(rec_labels, rec_predictions, average='micro'))

print
print
rec_conf_matrix = confusion_matrix(rec_labels, rec_predictions)
print "Confusion Matrix"
#print rec_conf_matrix
print
print "Prediction accuracy by label"
print "Labels     Species name       Recall Precis. F1-score"
print "-----------------------------------------------------"
for i in range(num_classes):
    TP = rec_conf_matrix[i][i]
    SUM = np.sum(rec_conf_matrix[i])
    if SUM == 0:
        recall = 0
    else:
        recall = float(TP)/SUM*100
    precision = float(TP)/np.sum(rec_conf_matrix[:,i])*100
    if recall == 0 and precision == 0:
        f1 = 0
    else:
        f1 = 2*(recall*precision)/(recall+precision)
    print "{:2d} {:^25} {:6.2f} | {:6.2f} | {:6.2f}".format(i, labels_dict[i], recall, precision, f1)


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

def accuracy(predictions, labels):
    return (1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

TPs = 0
FPs = 0
for i in range(len(file_predictions_mean)):
    if rec_labels[i] in file_predictions_top[i]:
        TPs += 1
    else:
        FPs += 1
        
print "Accuracy: " + str(float(TPs)/len(file_predictions_mean))
