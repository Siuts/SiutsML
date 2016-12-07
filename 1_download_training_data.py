import json
import urllib2
import urllib
import pickle
import csv
import os
from siuts import create_dir, Recording

data_dir = "data/"
create_dir(data_dir)

# Species are currently handpicked from PlutoF 
species_set = ['Parus_major', 'Coloeus_monedula', 'Corvus_cornix', 'Fringilla_coelebs',
               'Erithacus_rubecula', 'Phylloscopus_collybita', 'Turdus_merula', 'Cyanistes_caeruleus',
               'Emberiza_citrinella', 'Chloris_chloris', 'Turdus_philomelos', 'Phylloscopus_trochilus',
               'Sylvia_borin', 'Apus_apus', 'Passer_domesticus', 'Luscinia_luscinia', 'Sylvia_atricapilla',
               'Ficedula_hypoleuca', 'Sylvia_communis', 'Carpodacus_erythrinus']

acceptable_quality = ["A", "B"]

all_recordings = []
for species_name in species_set:
    species_split = species_name.split("_")
    genus = species_split[0]
    species = species_split[1]
    url = "http://www.xeno-canto.org/api/2/recordings?query={0}%20{1}".format(genus,species)
    json_data = json.load(urllib2.urlopen(url))
    recordings = []
    
    # if data is divided to several pages, then include them all
    page = int(json_data["page"])
    while (int(json_data["numPages"]) - page >= 0): 
        #creates list of Recordings objects
        quality_recordings = [Recording(x['id'], x['gen'], x['sp'], species_set.index("{0}_{1}".format(x["gen"], x["sp"])), x["file"]) for x in json_data["recordings"] if x["q"] in acceptable_quality]
        recordings = recordings + quality_recordings
        page += 1
        if (int(json_data["numPages"]) - page >= 0):
            json_data = json.load(urllib2.urlopen(url + "&page=" + str(page)))
    all_recordings = all_recordings + recordings
    
with open(data_dir+"training_recordings.pickle", 'wb') as f:
    pickle.dump(all_recordings, f, protocol=-1)
print "Finished downloading and saving training meta-data"

path = "data/mp3/"
create_dir(path)
recordings_count = len(all_recordings)

i = 0
for rec in all_recordings:
    file_path = path + rec.get_filename() + ".mp3"
    i = i + 1
    if i%100==0:
        print "{0}/{1} downloaded".format(i, recordings_count)

    if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
        urllib.urlretrieve (rec.file_url, path + rec.get_filename() +".mp3")
