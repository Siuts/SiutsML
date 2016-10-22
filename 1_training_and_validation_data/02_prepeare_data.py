import pickle
dataset_dir = "../data/"
with open(dataset_dir + "dataset_xeno.pickle", "rb") as datafile:
    dataset = pickle.load(datafile)


# ## Filter data
# Filter out good quality recordings and also the bird species not known (Myster mystery)

# In[2]:

acceptable_quality = ["A", "B"]
quality_recordings = [x for x in dataset if x["q"] in acceptable_quality and x["gen"] != "Mystery"]
print "Recordings: " + str(len(quality_recordings))


# In[3]:

import numpy as np
species = ["{0}_{1}".format(x["gen"], x["sp"]) for x in quality_recordings]
species_hist = dict((x, species.count(x)) for x in species)
print "Species list and histogram of filtered data generated"
print "Species: " + str(len(species_hist))


# In[4]:

sorted_species = sorted(species_hist, key=lambda x: species_hist[x], reverse=True)


# ## Reformat data
# * data = list([rec_id, file, label])
# * labels = list({label: species_name})

# In[5]:

labels = {}
labels_reverse = {}
for i in range(len(sorted_species)):
    labels[i] = sorted_species[i]
    labels_reverse[sorted_species[i]] = i
print "Labels dictionary created"
print "Labels: " + str(len(labels))


# In[6]:

data = []
rec_id = 0
for rec in quality_recordings:
    sp_name = "{0}_{1}".format(rec["gen"], rec["sp"])
    data.append([rec_id, rec["file"], labels_reverse[sp_name]])
    rec_id = rec_id + 1
print "Data list created"
print "Recordings: " + str(len(data))


import pickle
pickle.dump(data, open(dataset_dir + "dataset.pickle", "wb" ))
pickle.dump(labels, open(dataset_dir + "labels.pickle", "wb" ))
pickle.dump(labels_reverse, open(dataset_dir + "labels_reverse.pickle", "wb" ))

