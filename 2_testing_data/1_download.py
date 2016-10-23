
# coding: utf-8

# In[53]:

import json
import urllib2
import pickle
import urllib

download_dir = "../data/m4as"
dataset_dir = "../data/"
files_url_prefix = "https://files.plutof.ut.ee/"

with open("../data/labels.pickle", 'rb') as f:
    labels_dict = pickle.load(f)
print labels_dict
    
species = ["{0} {1}".format(x.split('_')[0], x.split('_')[1]) for x in labels_dict.values()]
print species


taxon_ids = {0: 86560, 1: 60814, 2: 107910, 3: 51922, 4: 56209, 5: 43289, 6: 88708, 7: 89511, 8: 71908, 9: 36397}

taxon_url_temp = "https://api.plutof.ut.ee/v1/taxonomy/taxonnodes/{0}/"
taxon_urls = { taxon_url_temp.format(v) : k for k, v in taxon_ids.items() }
print len(taxon_urls)


# In[47]:

recordings = []
counter = 0
url = "https://api.plutof.ut.ee/v1/public/taxonoccurrence/observation/observations/?mainform=15&page={}&page_size=100"

for page_nr in range(1, 17):
    json_data = json.load(urllib2.urlopen(url.format(page_nr)))
    print
    print "Downloading from page {}".format(page_nr)
    items = json_data['collection']['items']
    ####for item in items:....
    for item in items:
        links = item['links']
        taxon_url = [x['href'] for x in links if 'rel' in x and x['rel'] == 'taxon_node'][0]    
        if (taxon_url in taxon_urls.keys()):
            audio_urls = [x['href'] for x in links if 'format' in x and 'audio' in x['format']]
            if len(audio_urls) > 0:
                audio_url = audio_urls[0].replace("/public/", "/")
                audio_data = json.load(urllib2.urlopen(audio_url))
                file_loc = files_url_prefix + audio_data["public_url"]

                ext = file_loc.split(".")[-1]
                #print "{0} {1} {2}".format(counter, file_loc, ext)
                sp_name = labels_dict[taxon_urls[taxon_url]]
                fname = "{}_{:05d}".format(sp_name, audio_data["id"])
                urllib.urlretrieve(file_loc, "{}/{}.{}".format(download_dir, fname, ext))
                recordings.append([counter, fname, taxon_urls[taxon_url], ext])
                counter += 1
                if counter % 10 == 0:
                    print "Downloaded {} files".format(counter) 


# In[43]:

print recordings[:5]
with open("../data/test_dataset.pickle", 'wb') as f:
    pickle.dump(recordings, f, protocol=-1)
    
print len(recordings)

# In[54]:

from pydub import AudioSegment
import sys
errors = []
i = 0
for rec in recordings:
    try:
        i += 1
        print "{0}/{1}.m4a".format(download_dir, rec[1])
        sound = AudioSegment.from_file("{0}/{1}.{2}".format(download_dir, rec[1], rec[3]))

        sound = sound.set_frame_rate(22050).set_channels(1)
        sound.export("{0}test_wavs/{1}.wav".format(dataset_dir, rec[1]), format="wav")
    except:
        e = sys.exc_info()[0]
        print "Error on {0} - {1}".format(rec[0], rec[1])
        errors.append([i, rec[0], e])
    


# In[ ]:




# In[45]:

#print errors[0][2]

