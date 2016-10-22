
# coding: utf-8

# In[1]:

import pickle
import numpy as np
import random
import os

def find_biggest_in_dir(directory):
    all_files = os.listdir(directory)
    max_size = 0
    name = ""
    for fname in all_files:
        size = os.path.getsize(directory + fname)
        if size > max_size:
            max_size = size
            name = fname
    return name

uncut_dir = "../data/segments/1/training/"
save_dir = "../data/dataset/1/training/"


# In[2]:

with open(uncut_dir + find_biggest_in_dir(uncut_dir), 'rb') as f:
    max_segments = len(pickle.load(f))
    
print "Number of segments for each species: {0}".format(max_segments)


# In[3]:

with open("../data/labels.pickle", 'rb') as f:
    labels_dict = pickle.load(f)

species = labels_dict.values()
print species


# In[4]:

max_segments_in_file = 4096

for specimen in species:
    with open(uncut_dir + specimen + "_data.pickle", 'rb') as f:
        training_data = pickle.load(f)
    random.shuffle(training_data)
    nr_samples = len(training_data)
    if (nr_samples < max_segments):
        data_to_append = np.copy(training_data)
        for j in range(int(np.floor(max_segments/nr_samples))-1):
            training_data = np.concatenate((training_data, data_to_append))
        training_data = np.concatenate((training_data, data_to_append[:(max_segments-len(training_data))]))
    nr_of_files = int(np.ceil(float(max_segments)/max_segments_in_file))
    for i in range(nr_of_files):
        with open("{0}/{1}-training_{2}.pickle".format(save_dir, specimen, i), 'wb') as f:
            pickle.dump(training_data[i*max_segments_in_file:(i+1)*max_segments_in_file], f, protocol=-1)
    print specimen


# In[8]:

with open("species.pickle", 'wb') as f:
    pickle.dump(species, f, protocol=-1)

