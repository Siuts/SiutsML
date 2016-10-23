
# coding: utf-8

# In[1]:

import pickle
import operator
from os import listdir
from os.path import isfile, join
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import scale
import json

pickles_dir = "../data/segments/1/testing/"
num_labels = 10
image_size = 64

def load_pickled_segments_from_file_(filename, label, recId):
    data = load_java_segments_from_file(filename)
    if len(data) == 0:
        return np.empty([0]), np.empty([0]), np.empty([0])
    #print filename + " " + str(len(data))
    labels = []
    rec_Ids = []
    for _ in data:
        labels.append(label)
        rec_Ids.append(recId)
        segs = data
    return np.array(segs), reformat(np.array(labels)), np.array(rec_Ids)


def load_pickled_segments_from_file(filename, label, recId):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    if len(data) == 0:
        return np.empty([0]), np.empty([0]), np.empty([0])
    #print filename + " " + str(len(data))
    labels = []
    rec_Ids = []
    for _ in data:
        labels.append(label)
        rec_Ids.append(recId)
        segs = data
    return np.array(segs), reformat(np.array(labels)), np.array(rec_Ids)




def load_java_segments_from_file(filename):
    with open(filename) as data_file:    
        data = json.load(data_file)
        segments = []
        for jsonSegment in data:
            segment = []
            for jsonRow in jsonSegment:
                segment.append([float.fromhex(x) for x in jsonRow])
            segments.append(segment)
        return np.array(segments)
    
def scale_and_resize_segments(segments):
    segments = segments.reshape([len(segments), len(segments[0])*len(segments[0][0])])
    scaled_segments = scale(segments, axis=1, with_mean=True, with_std=True, copy=True )
    return scaled_segments.reshape(len(segments), image_size, image_size, 1)

def reformat(labels):
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return np.array(labels)


# In[2]:

with open('../data/test_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
    
test_files = [x.split(".")[0] for x in listdir(pickles_dir) if isfile(join(pickles_dir, x))]
print len(test_files)


# In[3]:

#print dataset


# # Create testing set

# In[4]:


counter = 0

all_segments = np.empty
all_labels = np.empty
all_rec_Ids = np.empty



for rec in dataset:
    fname = rec[1]
    label = rec[2]
    rec_id = rec[0]
    rec_segments, labels, rec_ids = load_pickled_segments_from_file(pickles_dir + fname + ".pickle", label, rec_id)
    if (rec_segments.shape[0] > 0 and labels.shape[0] > 0 and not np.isinf(np.sum(rec_segments))):
        processed_segments = scale_and_resize_segments(rec_segments)
        if counter == 0:
            all_segments = processed_segments
            all_labels = labels
            all_rec_Ids = rec_ids
        else:
            all_segments = np.vstack((all_segments, processed_segments))
            all_labels = np.vstack((all_labels, labels))
            all_rec_Ids = np.concatenate((all_rec_Ids, rec_ids))
        counter += 1
    if counter % 25 == 0:
        print str(counter) + "/" + str(len(test_files))


data_fname = "../data/dataset/1/testing/testing_data.pickle"
labels_fname = "../data/dataset/1/testing/testing_labels.pickle"
rec_Ids_fname = "../data/dataset/1/testing/testing_rec_ids.pickle"
    
with open(data_fname, 'wb') as f:
    pickle.dump(all_segments, f, protocol=-1)

with open(labels_fname, 'wb') as f:
    pickle.dump(all_labels, f, protocol=-1)

with open(rec_Ids_fname, 'wb') as f:
    pickle.dump(all_rec_Ids, f, protocol=-1)
print " "



# In[5]:

print all_labels[30]

