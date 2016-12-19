import numpy as np
import tensorflow as tf
import pickle
import matplotlib as mpl
import time
import sys
import siuts
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
mpl.use('Agg')

start = time.time()

num_epochs = 4 
num_files = 12
num_labels = 20


samples_in_file = 4096

dataset_loc = siuts.dataset_dir
test_dataset_length = 100


image_size = 64


batch_size = 128



def reformat(labels):
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return np.array(labels)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / len(predictions))

def load(fname):
    location = siuts.dataset_dir + fname + '.pickle'
    with open(location, 'rb') as f:
        images = pickle.load(f)
    #print(fname + " loaded!")
    return images


def load_labels(fname):
    with open(dataset_loc + fname + '.pickle', 'rb') as f:
        labels = pickle.load(f)
    return labels


print("Functions loaded!")

graph_path = "checkpoints/"

species = siuts.species_list
        
validation_dataset = load("validation_data")

print("Datasets loaded")
validation_labels = reformat(load("validation_labels"))
print("Labels loaded")

num_channels = 1  # grayscale
valid_dataset = validation_dataset
valid_labels = validation_labels

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
    
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)




    
    def conv2d(name, data, kernel_shape, bias_shape, stride=1):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(0.0, 0.05))
            biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(1.0))
            
            conv = tf.nn.conv2d(data, weights, [1, stride, stride, 1], padding='SAME', name="conv")
            pre_activation = tf.nn.bias_add(conv, biases)
            activation = tf.nn.elu(pre_activation, name="elu")
            print activation
            _activation_summary(activation)
            return activation
        
    def fully_connected(name, data, weights_shape, bias_shape, dropout):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable("weights", weights_shape, initializer=tf.random_normal_initializer(0.0, 0.05))
            biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(1.0))
            activation = tf.nn.elu(tf.nn.bias_add(tf.matmul(data, weights),  biases), name="elu")
            print activation
            _activation_summary(activation)
            return tf.nn.dropout(activation, dropout, name="dropout")
        
    def _activation_summary(x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                                           tf.nn.zero_fraction(x))

    # Model.
    def model(data, input_dropout, fc_dropout):
        data = tf.nn.dropout(data, input_dropout)
        # Conv1
        conv = conv2d("conv1", data, [5 , 5, 1, 32], [32], 2)
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')


        # Conv2
        conv = conv2d("conv2", pool, [5, 5, 32, 96], [96])
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')


        # Conv3
        conv = conv2d("conv3", pool, [3, 3, 96, 128], [128])
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
         
        # Conv4
        conv = conv2d("conv4", pool, [3, 3, 128, 256], [256])
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        
        # Conv5
#        conv = conv2d("conv5", pool, [3, 3, 192, 256], [256])
#        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

#         # Conv1
#         conv = conv2d("conv1", data, [5 , 5, 1, 32], [32], 2)
#         pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')


#         # Conv2
#         conv = conv2d("conv2", pool, [4, 4, 32, 96], [96])
#         pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')


#         # Conv3
#         conv = conv2d("conv3", pool, [3, 3, 96, 128], [128])
#         pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # Fully connected1
        shape = pool.get_shape().as_list()
        reshaped_layer = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        
        fc = fully_connected("fc1", reshaped_layer, [shape[1] * shape[2] * shape[3], 256], [256], fc_dropout)

        # Fully connected 2
        fc = fully_connected("fc2", fc, [256, 128], [128], fc_dropout)

        # output layer
        return fully_connected("output", fc, [128, num_labels], [num_labels], 1)

    # Training computation.
    logits = model(tf_train_dataset, 1, 0.8)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits + 1e-50, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.MomentumOptimizer(0.005, 0.9, use_locking=False, name='Momentum', use_nesterov=True).minimize(loss)
    #optimizer = tf.train.MomentumOptimizer(0.01, 0.9, use_locking=False, name='Momentum', use_nesterov=True).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

    # Predictions for the training, validation, and test data.
    tf.get_variable_scope().reuse_variables()
    train_prediction = tf.nn.softmax(model(tf_train_dataset, 1, 1), name="sm_train")
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1, 1), name="sm_valid")
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 1, 1), name="sm_test")
    one_prediction = tf.nn.softmax(model(tf_one_prediction, 1, 1), name="sm_one")


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
                new_data = load("{0}-training_{1}".format(specimen, current_file))
                new_labels = np.empty(new_data.shape[0])
                new_labels.fill(siuts.species_list.index(specimen))
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


x_axis = np.arange(0.0, step, 25.0)
plt.plot(x_axis, training_acc, "r", x_axis, validation_acc, "b")
plt.savefig(graph_path + "batch_vs_eval_acc.png")
print("Training took " + str(time.time() - start) + " seconds")
print("Training done!")