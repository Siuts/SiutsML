import json
import urllib2

# Species are currently handpicked from PlutoF 
species_set = ['Parus_major', 'Coloeus_monedula', 'Corvus_cornix', 'Fringilla_coelebs',
               'Erithacus_rubecula', 'Phylloscopus_collybita', 'Turdus_merula', 'Cyanistes_caeruleus',
               'Emberiza_citrinella', 'Chloris_chloris', 'Turdus_philomelos', 'Phylloscopus_trochilus',
               'Sylvia_borin', 'Apus_apus', 'Passer_domesticus', 'Luscinia_luscinia', 'Sylvia_atricapilla',
               'Ficedula_hypoleuca', 'Sylvia_communis', 'Carpodacus_erythrinus']


# ### JSON for all the species recordings

# In[2]:

all_recordings = []
for species_name in species_set:
    species_split = species_name.split("_")
    genus = species_split[0]
    species = species_split[1]
    url = "http://www.xeno-canto.org/api/2/recordings?query={0}%20{1}".format(genus,species)
    json_data = json.load(urllib2.urlopen(url))
    recordings = json_data["recordings"]
    
    # if data is divided to several pages, then include them all
    page = 1
    while (int(json_data["numPages"]) - page != 0): 
        page = page + 1
        json_data = json.load(urllib2.urlopen(url + "&page=" + str(page)))
        recordings = recordings + json_data["recordings"]
        
    all_recordings.append(recordings)
print "Finished downloading data on all species"


# In[3]:

all_recordings_dict = dict(zip(species_set, all_recordings))
all_recordings_hist = dict((x, len(y)) for x, y in all_recordings_dict.items()) 
print "Dictionary of species and recordings is made"


# ### Download and generate metadata in csv and pickle format

# In[4]:

import urllib
import pickle
import csv
import os

download_dir = "../data/"

cv_titles = ["file", "id", "gen", "sp", "ssp", "en", "rec", "cnt", "loc", "lat", "lng", "type", "lic", "url", "q", "time", "date"]
cv_rows = []
dataset = []

i = 0
for sp_recordings in all_recordings:
    for rec in sp_recordings:
        # filename in format of "latin_name-id.mp3"
        filename = "{0}_{1}-{2}".format(rec["gen"], rec["sp"], rec["id"])
        path = download_dir + "mp3/"
        file_path = path + filename + ".mp3"
        i = i + 1
        if i%100==0:
            print "{0}/{1} downloaded".format(i,sum(all_recordings_hist.values()))
            
        # only download if file doesn't exist or it's size is 0
        if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
            urllib.urlretrieve (rec["file"], path + filename+".mp3")
            
        cv_row = []
        for key in cv_titles:
            if key=="file":
                cv_row.append(filename)
            else:
                if (rec[key] is None):
                    cv_row.append("")
                else:
                    cv_row.append(rec[key].encode('utf8'))
        cv_rows.append(cv_row)

        dataset_row = dict(rec)
        dataset_row["file"] = filename
        dataset.append(dataset_row)

pickle.dump(dataset, open(download_dir+"dataset_xeno.pickle", "wb" ))
pickle.dump(species_set, open(download_dir+"species_list.pickle", "wb" ))

with open(download_dir+"dataset_xeno.csv", 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
    writer.writerow(cv_titles)
    for row in cv_rows:
        writer.writerow(row)

print "Dataset saved!"
