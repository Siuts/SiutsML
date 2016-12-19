import siuts
from siuts import create_dir, Recording
import time
import pickle
import os
import numpy as np
import scipy.misc

def segment_wavs(recordings_file, segments_dir, wavs_dir):
    create_dir(segments_dir)
    with open(recordings_file, "rb") as f:
        recordings = pickle.load(f)
    recordings_count = len(recordings)
    for counter, rec in enumerate(recordings):
        fname = rec.get_filename()
        #print fname
        pickle_path = segments_dir + fname + ".pickle"
        if (not os.path.isfile(pickle_path)):
            wav_path = "{0}{1}.wav".format(wavs_dir, fname)
            if (os.path.isfile(wav_path)):
                signal, fs = siuts.get_wav_info(wav_path)
                transposed_spectrogram = abs(siuts.stft(signal, siuts.fft_frame_size))[:,:siuts.fft_frame_size/2]
                cleaned_spectrogram = siuts.clean_spectrogram(transposed_spectrogram)
                if cleaned_spectrogram.shape[0] > siuts.fft_frame_size/2:
                    segments = []
                    hop_size = siuts.segmentation_hop_size

                    for i in range(int(np.floor(cleaned_spectrogram.shape[0] / hop_size - 1))):
                        segment = cleaned_spectrogram[i * hop_size:i * hop_size + cleaned_spectrogram.shape[1]]
                        resized_segment = scipy.misc.imresize(segment, (siuts.resized_segment_size, siuts.resized_segment_size), interp='nearest')
                        segments.append(resized_segment)
                    with open(segments_dir + fname + ".pickle", 'wb') as f:
                        pickle.dump(segments, f, protocol=-1)
        if (counter % 100 == 0):
            print "{0}/{1} file segmented".format(counter, recordings_count)
            
print "Starting training data segmentation"
start = time.time()
segment_wavs(siuts.training_recordings_path, siuts.training_segments_dir, siuts.training_wavs_dir)
print "Training data segmentation took {0} seconds".format(time.time() - start)

print "Starting testing segmentation"
start = time.time()
segment_wavs(siuts.testing_recordings_path, siuts.testing_segments_dir, siuts.testing_wavs_dir)
print "Testing data segmentation took {0} seconds".format(time.time() - start)
    
