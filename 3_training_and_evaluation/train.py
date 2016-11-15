import numpy as np
import tensorflow as tf
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import time
import sys

start = time.time()

num_epochs = 2 
samples_in_file = 4096
num_files = 26

dataset_loc = '../data/dataset/1/'
test_dataset_length = 100


image_size = 64
num_labels = 20

batch_size = 128



def reformat(labels):
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return np.array(labels)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def load(fname):
    location = dataset_loc + fname + '.pickle'
    with open(location, 'rb') as f:
        images = pickle.load(f)
    #print(fname + " loaded!")
    return images


def load_labels(fname):
    with open(dataset_loc + fname + '.pickle', 'rb') as f:
        labels = pickle.load(f)
    return labels


print("Functions loaded!")

graph_path = "logs/"

with open("../data/labels_reverse.pickle", 'rb') as f:
    labels_dict = pickle.load(f)
    
#with open(dataset_loc + "../../data/datasets/1/species.pickle", 'rb') as f:
#    species = pickle.load(f)
species = labels_dict.keys()

print species

        
validation_dataset = load("validation/validation_data")

print("Datasets loaded")
validation_labels = load("validation/validation_labels")
print("Labels loaded")

num_channels = 1  # grayscale
valid_dataset = validation_dataset
valid_labels = validation_labels




patch1_size = 8
patch2_size = 4
patch3_size = 3
patch4_size = 3
patch5_size = 3
conv_depth1 = 32
conv_depth2 = 96
conv_depth3 = 128
conv_depth4 = 128
conv_depth5 = 128
num_hidden1 = 256
num_hidden2 = 128

conv_stride = 2

graph = tf.Graph()

with graph.as_default():
    tf.set_random_seed(1337)
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels), name="train_dataset_placeholder")
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name="train_labels_placeholder")
    tf_valid_dataset = tf.placeholder(tf.float32, shape=(valid_dataset.shape[0], image_size, image_size, num_channels), name="valid_dataset_placeholder")
    tf_test_dataset = tf.placeholder(tf.float32, shape=(test_dataset_length, image_size, image_size, num_channels),
                                     name="test_dataset_placeholder")
    tf_one_prediction = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels),
                                       name="tf_one_prediction")

    # Variables.
    # Conv 1 + Max pool 1
    conv1_kernel = tf.Variable(tf.truncated_normal(
        [patch1_size, patch1_size, num_channels, conv_depth1], stddev=0.1), name="conv1_kernel")
    conv1_biases = tf.Variable(tf.zeros([conv_depth1]), name="conv1_biases")

    # Conv 2 + Max pool 2
    conv2_kernel = tf.Variable(tf.truncated_normal(
        [patch2_size, patch2_size, conv_depth1, conv_depth2], stddev=0.1), name="conv2_kernel")
    conv2_biases = tf.Variable(tf.constant(1.0, shape=[conv_depth2]), name="conv2_biases")

    # Conv 3 + Max pool 3
    conv3_kernel = tf.Variable(tf.truncated_normal(
        [patch3_size, patch3_size, conv_depth2, conv_depth3], stddev=0.1), name="conv3_kernel")
    conv3_biases = tf.Variable(tf.constant(1.0, shape=[conv_depth3]), name="conv3_biases")

    # Conv 4 + Max pool 4
    conv4_kernel = tf.Variable(tf.truncated_normal(
        [patch4_size, patch4_size, conv_depth3, conv_depth4], stddev=0.1), name="conv4_kernel")
    conv4_biases = tf.Variable(tf.constant(1.0, shape=[conv_depth4]), name="conv4_biases")

    # Conv 5 + Max pool 5
    conv5_kernel = tf.Variable(tf.truncated_normal(
        [patch5_size, patch5_size, conv_depth4, conv_depth5], stddev=0.1), name="conv5_kernel")
    conv5_biases = tf.Variable(tf.constant(1.0, shape=[conv_depth5]), name="conv5_biases")

    # Fully connected
    fc1_weights = tf.Variable(tf.truncated_normal([2048, num_hidden1], stddev=0.1), name="fc1_weights")
    fc1_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name="fc1_biases")

    fc2_weights = tf.Variable(tf.truncated_normal(
        [num_hidden1, num_hidden2], stddev=0.1), name="fc2_weights")
    fc2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]), name="fc2_biases")

    fc3_weights = tf.Variable(tf.truncated_normal(
        [num_hidden2, num_labels], stddev=0.1), name="fc3_weights")
    fc3_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name="fc3_biases")


    # Model.
    def training_model(data):
        print("Data shape: " + str(data.get_shape()))

        # Conv1
        conv = tf.nn.conv2d(data, conv1_kernel, [1, conv_stride, conv_stride, 1], padding='SAME', name="conv1")
        hidden = tf.nn.elu(conv + conv1_biases, name="relu1")
        print "Conv1 shape: " + str(hidden.get_shape())

        hidden = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')
        #hidden = tf.nn.dropout(hidden, 0.9, name="dropout1")
        print "Max pool 1 shape: " + str(hidden.get_shape())

        # Conv2
        conv = tf.nn.conv2d(hidden, conv2_kernel, [1, 1, 1, 1], padding='SAME', name="conv2")
        hidden = tf.nn.elu(conv + conv2_biases, name="relu2")
        print "Conv2 shape: " + str(hidden.get_shape())

        hidden = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')
        #hidden = tf.nn.dropout(hidden, 0.8, name="dropout2")
        print "Max pool 2 shape: " + str(hidden.get_shape())

        # Conv3
        conv = tf.nn.conv2d(hidden, conv3_kernel, [1, 1, 1, 1], padding='SAME', name="conv3")
        hidden = tf.nn.elu(conv + conv3_biases, name="relu3")
        print "Conv3 shape: " + str(hidden.get_shape())

        # Conv4
        #conv = tf.nn.conv2d(hidden, conv4_kernel, [1, 1, 1, 1], padding='SAME', name="conv4")
        #hidden = tf.nn.elu(conv + conv4_biases, name="relu4")
        #print "Conv4 shape: " + str(hidden.get_shape())

        # Conv5
        #conv = tf.nn.conv2d(hidden, conv5_kernel, [1, 1, 1, 1], padding='SAME', name="conv5")
        #hidden = tf.nn.elu(conv + conv5_biases, name="relu5")
        #print "Conv5 shape: " + str(hidden.get_shape())

        hidden = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        #hidden = tf.nn.dropout(hidden, 0.75, name="dropout2")
        print "Max pool 5 shape: " + str(hidden.get_shape())

        # Fully connected
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.elu(tf.matmul(reshape, fc1_weights) + fc1_biases, name="relu6")
        hidden = tf.nn.dropout(hidden, 0.75, name="dropout3")
        print "Fully connected 1 shape: " + str(hidden.get_shape())

        hidden = tf.nn.elu(tf.matmul(hidden, fc2_weights) + fc2_biases, name="relu7")
        hidden = tf.nn.dropout(hidden, 0.75, name="dropout4")
        print "Fully connected 2 shape: " + str(hidden.get_shape())

        # Fully connected
        return tf.matmul(hidden, fc3_weights) + fc3_biases


    # Model.
    def model(data):
        print("Data shape: " + str(data.get_shape()))

        # Conv1
        conv = tf.nn.conv2d(data, conv1_kernel, [1, conv_stride, conv_stride, 1], padding='SAME', name="conv1")
        hidden = tf.nn.elu(conv + conv1_biases, name="relu1")
        hidden = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')

        # Conv2
        conv = tf.nn.conv2d(hidden, conv2_kernel, [1, 1, 1, 1], padding='SAME', name="conv2")
        hidden = tf.nn.elu(conv + conv2_biases, name="relu2")
        hidden = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')

        # Conv3
        conv = tf.nn.conv2d(hidden, conv3_kernel, [1, 1, 1, 1], padding='SAME', name="conv3")
        hidden = tf.nn.elu(conv + conv3_biases, name="relu3")

        # Conv4
        #conv = tf.nn.conv2d(hidden, conv4_kernel, [1, 1, 1, 1], padding='SAME', name="conv4")
        #hidden = tf.nn.elu(conv + conv4_biases, name="relu4")

        # Conv5
        #conv = tf.nn.conv2d(hidden, conv5_kernel, [1, 1, 1, 1], padding='SAME', name="conv5")
        #hidden = tf.nn.elu(conv + conv5_biases, name="relu5")

        hidden = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')

        # Fully connected
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.elu(tf.matmul(reshape, fc1_weights) + fc1_biases, name="relu6")

        hidden = tf.nn.elu(tf.matmul(hidden, fc2_weights) + fc2_biases, name="relu7")

        # Fully connected
        return tf.matmul(hidden, fc3_weights) + fc3_biases


    # Training computation.
    logits = training_model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits + 1e-50, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(model(tf_train_dataset), name="sm_train")
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset), name="sm_valid")
    test_prediction = tf.nn.softmax(model(tf_test_dataset), name="sm_test")
    one_prediction = tf.nn.softmax(model(tf_one_prediction), name="sm_one")

# In[6]:


counter = 0
old_loss = 100000
max_eval_acc = 1
training_acc = []
validation_acc = []
checkpoint_path = graph_path + "model.ckpt"

with tf.Session(graph=graph) as session:
    writer = tf.train.SummaryWriter(graph_path, session.graph)
    tf.initialize_all_variables().run()
    print('Initialized')
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=30)
    tf.train.write_graph(session.graph_def, graph_path, "graph.pb", False)  # proto
    
    train_dataset = np.empty
    train_labels = np.empty
    current_file = 0
    current_epoch = 1
    step = 0
    while True:
        if (step*batch_size) % (samples_in_file * num_labels - batch_size) == 0:
            if current_epoch > num_epochs:
            	break
            del train_dataset
            del train_labels
            sys.stdout.write("Loading datasets nr " + str(current_file))
            sys.stdout.flush()
            counter = 0
			
            train_dataset = np.empty
            train_labels = np.empty
            for specimen in species[:num_labels]:
                new_data = load("training/{0}-training_{1}".format(specimen, current_file))
                new_labels = np.empty(new_data.shape[0])
                new_labels.fill(labels_dict[specimen])
                if counter == 0:
                    train_dataset = new_data
                    train_labels = new_labels
                else:
                    train_dataset = np.vstack((train_dataset, new_data))
                    train_labels = np.concatenate((train_labels, new_labels))
                counter += 1
                #print train_dataset.shape
                sys.stdout.write(".")
                sys.stdout.flush()
            del new_data
            print ""
            current_file += 1
            if current_file >= num_files:
                current_file = 0
                current_epoch += 1
            train_dataset, _, train_labels, _ = train_test_split(train_dataset, reformat(train_labels), test_size=0, random_state=1337)
        offset = (step * batch_size) % (num_labels*samples_in_file - batch_size)
        sys.stdout.write(".")
        sys.stdout.flush()
        
    
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 25 == 0:
            batch_acc = accuracy(predictions, batch_labels)
            eval_acc = accuracy(valid_prediction.eval(feed_dict={tf_valid_dataset: valid_dataset}), valid_labels)
            training_acc.append(batch_acc)
            validation_acc.append(eval_acc)

            if step % 1000 == 0:
                saver.save(session, checkpoint_path, global_step=step)

            print('%d - Minibatch loss: %f | Minibatch accuracy: %.1f%% | Validation accuracy: %.1f%%' % (
                step, l, batch_acc, eval_acc))
        step += 1

    saver.save(session, checkpoint_path, global_step=step)
    del train_dataset

# In[ ]:
x_axis = np.arange(0.0, step, 25.0)
plt.plot(x_axis, training_acc, "r", x_axis, validation_acc, "b")
plt.savefig(graph_path + "batch_vs_eval_acc.png")
print("Training took " + str(time.time() - start) + " seconds")
print("Training done!")
