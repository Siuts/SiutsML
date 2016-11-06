import pickle
import os
from os import listdir
from os.path import isfile, join
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import scale
import random

pickles_dir = "../data/segments/1/training/"

joined_training_dir = "../data/segments/1/training_joined/"
save_dir = "../data/dataset/1/training/"
val_dir = "../data/dataset/1/validation/"
validation_data_fname = val_dir + "validation_data.pickle"
validation_labels_fname = val_dir +"validation_labels.pickle"
validation_rec_Ids_fname = val_dir + "validation_rec_ids.pickle"

num_labels = 20
image_size = 64

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
    
def scale_and_resize_segments(segments):
    segments = segments.reshape([len(segments), len(segments[0])*len(segments[0][0])])
    scaled_segments = scale(segments, axis=1, with_mean=True, with_std=True, copy=True )
    return scaled_segments.reshape(len(segments), image_size, image_size, 1)

def reformat(labels):
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return np.array(labels)


# In[2]:

with open('../data/dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

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

with open(pickles_dir + find_biggest_in_dir(pickles_dir), 'rb') as f:
    max_segments = len(pickle.load(f))
    
recordings = [x.split(".")[0] for x in listdir(pickles_dir) if isfile(join(pickles_dir, x))]
train_files, valid_files = train_test_split(recordings, test_size=0.1, random_state=23)
species = list(set([x.split("-")[0] for x in train_files]))

print "Total amount of recordings: {0}".format(len(recordings))
print "Total amount of training files: {0}".format(len(train_files))

counter = 0

# list of rec_ids, filenames and labels from training/validation set and corresponding to specific species
validation_files = [x for x in dataset if x[1] in valid_files]
all_segments = np.empty
all_labels = np.empty
all_rec_Ids = np.empty
validation_segments_count = {}
# if not isfile(validation_data_fname):
#     for rec in validation_files:
#         fname = rec[1]
#         label = rec[2]
#         rec_id = rec[0]
#         rec_segments, labels, rec_ids = load_pickled_segments_from_file(pickles_dir + fname + ".pickle", label, rec_id)
#         if (rec_segments.shape[0] > 0 and labels.shape[0] > 0 and not np.isinf(np.sum(rec_segments))):
#             processed_segments = scale_and_resize_segments(rec_segments)
#             if counter == 0:
#                 all_segments = processed_segments
#                 all_labels = labels
#                 all_rec_Ids = rec_ids
#             else:
#                 all_segments = np.vstack((all_segments, processed_segments))
#                 all_labels = np.vstack((all_labels, labels))
#                 all_rec_Ids = np.concatenate((all_rec_Ids, rec_ids))
#             specimen = fname.split("-")[0]
#             if specimen in validation_segments_count:
#                 validation_segments_count[specimen] = validation_segments_count[specimen] + all_segments.shape[0]
#             else:
#                 validation_segments_count[specimen] = all_segments.shape[0]
#         if counter % 25 == 0:
#             print str(counter) + "/" + str(len(validation_files))
#         counter += 1
#
#     with open(validation_data_fname, 'wb') as f:
#         pickle.dump(all_segments, f, protocol=-1)
#
#     with open(validation_labels_fname, 'wb') as f:
#         pickle.dump(all_labels, f, protocol=-1)
#
#     with open(validation_rec_Ids_fname, 'wb') as f:
#         pickle.dump(all_rec_Ids, f, protocol=-1)
all_segments = []
all_labels = []
all_rec_Ids = []
if not isfile(validation_data_fname):
    for rec in validation_files:
        fname = rec[1]
        label = rec[2]
        rec_id = rec[0]
        rec_segments, labels, rec_ids = load_pickled_segments_from_file(pickles_dir + fname + ".pickle", label, rec_id)
        if (rec_segments.shape[0] > 0 and labels.shape[0] > 0 and not np.isinf(np.sum(rec_segments))):
            processed_segments = scale_and_resize_segments(rec_segments)
            all_segments = all_segments + processed_segments.tolist()
            all_labels = all_labels + labels.tolist()
            all_rec_Ids = all_rec_Ids + rec_ids.tolist()

            specimen = fname.split("-")[0]
            if specimen in validation_segments_count:
                validation_segments_count[specimen] = validation_segments_count[specimen] + len(all_segments)
            else:
                validation_segments_count[specimen] = len(all_segments)
        if counter % 25 == 0:
            print str(counter) + "/" + str(len(validation_files))
        counter += 1
    all_segments = np.array(all_segments)
    all_labels = np.array(all_labels)
    all_rec_Ids = np.array(all_rec_Ids)
    with open(validation_data_fname, 'wb') as f:
        pickle.dump(all_segments, f, protocol=-1)

    with open(validation_labels_fname, 'wb') as f:
        pickle.dump(all_labels, f, protocol=-1)

    with open(validation_rec_Ids_fname, 'wb') as f:
        pickle.dump(all_rec_Ids, f, protocol=-1)
print validation_segments_count



# In[7]:

## with open('../../labels_reverse.pickle', 'rb') as f:
#     t = pickle.load(f)
# print dataset

max_segments_in_file = 4096

dataset_name = "training"
files_set = train_files
training_segments_count = {}
for specimen in species:
    # list of rec_ids, filenames and labels from training/validation set and corresponding to specific species
    specimen_files = [x for x in dataset if x[1].split("-")[0] == specimen and x[1] in files_set]
    all_segments = np.empty
    all_labels = np.empty
    all_rec_Ids = np.empty

    filepath_prefix = "{0}{1}_".format(pickles_dir, specimen)
    data_fname = filepath_prefix + "data.pickle"
    labels_fname = filepath_prefix + "labels.pickle"
    rec_Ids_fname = filepath_prefix + "rec_ids.pickle"

    counter = 0
    print specimen + " " + str(len(specimen_files))
    if not (isfile(data_fname) and isfile(labels_fname) and isfile(rec_Ids_fname)):
        for rec in specimen_files:
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
                print str(counter) + "/" + str(len(specimen_files))
        print all_segments.shape

        training_data = all_segments
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
        print specimen + " saved"
        training_segments_count[specimen] = training_data.shape[0]






        #with open(data_fname, 'wb') as f:
        #    pickle.dump(all_segments, f, protocol=-1)

        with open(labels_fname, 'wb') as f:
            pickle.dump(all_labels, f, protocol=-1)

        with open(rec_Ids_fname, 'wb') as f:
            pickle.dump(all_rec_Ids, f, protocol=-1)
        print " "

print training_segments_count
