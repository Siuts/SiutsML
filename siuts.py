import os


# List of species used in classification task. The index of the list is the label for each species
species_list = ['Parus_major', 'Coloeus_monedula', 'Corvus_cornix', 'Fringilla_coelebs',
               'Erithacus_rubecula', 'Phylloscopus_collybita', 'Turdus_merula', 'Cyanistes_caeruleus',
               'Emberiza_citrinella', 'Chloris_chloris', 'Turdus_philomelos', 'Phylloscopus_trochilus',
               'Sylvia_borin', 'Apus_apus', 'Passer_domesticus', 'Luscinia_luscinia', 'Sylvia_atricapilla',
               'Ficedula_hypoleuca', 'Sylvia_communis', 'Carpodacus_erythrinus']

# PlutoF species ID-s had to be handpicked, because the names didn't always correspond to the ones in Xeno-Canto. Each ID in this list corresponds to the species in species_list
plutoF_taxon_ids = [86560, 48932, 110936, 60814, 57887, 89499, 107910, 86555, 56209, 43289, 107914, 89514, 102321, 36397, 86608, 72325, 102319, 60307, 102323, 43434]

# xeno-canto quality A is the best
acceptable_quality = ["A", "B"]

wav_framerate = 22050

fft_frame_size = 512

resized_segment_size = 64

# overlap by half of the segment size
segmentation_hop_size = fft_frame_size/4

data_dir = "data/"
xeno_dir = data_dir + "xeno_recordings/"
plutoF_dir = data_dir + "plutof_recordings/"
training_recordings_path = data_dir + "training_recordings.pickle"
testing_recordings_path = data_dir + "testing_recordings.pickle"
training_wavs_dir = data_dir + "training_wavs/"
testing_wavs_dir = data_dir + "testing_wavs/"
training_segments_dir = data_dir + "training_segments/"
testing_segments_dir = data_dir + "testing_segments/"

dataset_dir = data_dir + "dataset/"
testing_data_filepath = dataset_dir + "testing_data.pickle"
testing_labels_filepath = dataset_dir + "testing_labels.pickle"
testing_rec_ids_filepath = dataset_dir + "testing_rec_ids.pickle"
validation_data_filepath = dataset_dir + "validation_data.pickle"
validation_labels_filepath = dataset_dir + "validation_labels.pickle"
validation_rec_ids_filepath = dataset_dir + "validation_rec_ids.pickle"

class Recording:
    def __init__(self, id, gen, sp, label, file_url):
        self.id = id
        self.gen = gen
        self.sp = sp
        self.label = label
        self.file_url = file_url
        
    def __repr__(self):
        return "id: {0}, name: {1}_{2}, label: {3}".format(self.id, self.gen, self.sp, self.label)
        
    def get_name(self):
        """Return the scientific name - <genus_species>"""
        return "{0}_{1}".format(self.gen, self.sp)
        
    def get_filename(self):
        """Return the filename withoud extension - <genus_species-id>"""
        return "{0}_{1}-{2}".format(self.gen, self.sp, self.id)
    
def create_dir(path):
    (dirname, _) = os.path.split(path)
    if (not os.path.isdir(dirname)):
        os.makedirs(dirname)

import wave
import pylab
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

import numpy as np
from numpy.lib import stride_tricks
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1

    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.fft(frames)  

def clean_spectrogram(transposed_spectrogram, coef=3):
    row_means = transposed_spectrogram.mean(axis=0)
    col_means = transposed_spectrogram.mean(axis=1)
    
    cleaned_spectrogram = []

    for col_index, column in enumerate(transposed_spectrogram):
        for row_index, pixel in enumerate(column):
            if (pixel > coef*row_means[row_index] and pixel > coef*col_means[col_index]):
                cleaned_spectrogram.append(transposed_spectrogram[col_index])
                break
    return np.array(cleaned_spectrogram)

from sklearn.preprocessing import scale
def scale_segments(segments):
    segment_size = len(segments[0])
    segment_count = len(segments)
    segments = segments.reshape([segment_count, segment_size*segment_size])
    scaled_segments = scale(segments, axis=1, with_mean=True, with_std=True, copy=True )
    return scaled_segments.reshape(segment_count, segment_size, segment_size, 1)

