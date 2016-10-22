
# coding: utf-8

# In[1]:

import pickle
from pydub import AudioSegment
import datetime
import sys
import time

start = time.time()

dataset_dir = "../data/"

with open(dataset_dir + "dataset.pickle", "rb") as datafile:
    dataset = pickle.load(datafile)


# In[2]:

errors = []
i=0
for rec in dataset:
    if i%1000 == 0:
        print "{0}/{1} | {2}".format(i, len(dataset), str(datetime.datetime.now()))
    i += 1
    try:
        sound = AudioSegment.from_mp3("{0}mp3/{1}.mp3".format(dataset_dir, rec[1]))
        sound = sound.set_frame_rate(22050).set_channels(1)
        sound.export("{0}wavs/{1}.wav".format(dataset_dir, rec[1]), format="wav")
    except:
        e = sys.exc_info()[0]
        print "Error on {0} - {1}".format(rec[0], rec[1])
        errors.append([i, rec[0], e])

end = time.time()    
print "Converting from mp3 to wav took {0} seconds".format(end - start)

