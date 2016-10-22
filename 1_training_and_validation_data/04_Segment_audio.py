import numpy as np
import pandas as pd
import scipy as sp
import  pickle
from scipy import fft
from time import localtime, strftime
from skimage.morphology import  disk,remove_small_objects
from skimage.filter import rank
from skimage.util import img_as_ubyte 
import wave
import pylab
from numpy.lib import stride_tricks
import matplotlib.patches as patches
from os import listdir
from os.path import isfile, join
import scipy
import os

###########################
# Folder Name Setting
###########################
folder = '../data/'
segments_folder = folder + 'segments/1/training/'
wav_folder = folder + "wavs/"
num_species = 5

recordings = pickle.load(open(folder + "dataset.pickle", "rb"))


FFT_FRAME_SIZE = 512
FFT_FRAME_RES = 256
MIN_SEGMENT_SIZE = 400
P = 90 #percentange in binary 
FRAME_RATE = 22050


SEGMENT_SIZE = 64

SAVE_PLOT = False
USE_LOG_SPECTOGRAMS = False

def segment_spectogram(mypic_rev, min_segment_size, p, sigma=3, grad_disk=3,):
    mypic_rev_gauss = sp.ndimage.gaussian_filter(mypic_rev, sigma=sigma)    
    mypic_rev_gauss_bin = mypic_rev_gauss > np.percentile(mypic_rev_gauss,p)    
    mypic_rev_gauss_bin_close =sp.ndimage.binary_closing( sp.ndimage.binary_opening(mypic_rev_gauss_bin))    
    mypic_rev_gauss_grad = rank.gradient(pic_to_ubyte(mypic_rev_gauss), disk(grad_disk))
    mypic_rev_gauss_grad_bin = mypic_rev_gauss_grad > np.percentile(mypic_rev_gauss_grad,p)  
    mypic_rev_gauss_grad_bin_close =sp.ndimage.binary_closing( sp.ndimage.binary_opening(mypic_rev_gauss_grad_bin))    
    bfh = sp.ndimage.binary_fill_holes(mypic_rev_gauss_grad_bin_close)        
    bfh_rm = remove_small_objects(bfh, min_segment_size)
    return sp.ndimage.label(bfh_rm)



def get_segments(fname, spectogram):
    SPEC_SEGMENTS = []
    big_ROIs = 0

    mypic_rev = fft
    if not USE_LOG_SPECTOGRAMS: 
        labeled_segments, num_seg = segment_spectogram(spectogram, MIN_SEGMENT_SIZE, P)


    if USE_LOG_SPECTOGRAMS:
        spectogram_log = np.log10(spectogram+ 0.001)
        labeled_segments, num_seg = segment_spectogram(spectogram_log, MIN_SEGMENT_SIZE, P)

    if SAVE_PLOT:
        fig = plt.figure() 
        plot = fig.add_subplot(111, aspect='equal')
        plot.imshow(spectogram)
        plot.axis('off')
        
    not_allowed_centers_list = []
    for current_segment_id in range(1,num_seg+1):
        current_segment = (labeled_segments == current_segment_id)*1
        xr = current_segment.max(axis =  1)
        yr = current_segment.max(axis =  0)
        xr_max = np.max(xr*np.arange(len(xr)))
        xr[xr==0] = xr.shape[0]
        xr_min = np.argmin(xr)         
        yr_max = np.max(yr*np.arange(len(yr)))
        yr[yr==0] = yr.shape[0]
        yr_min = np.argmin(yr)
        xr_width = xr_max-xr_min
        if xr_width > FFT_FRAME_RES:
            big_ROIs += 1
            if SAVE_PLOT: 
                plot.add_patch(patches.Rectangle((xr_min, yr_min), (xr_max-xr_min), (yr_max-yr_min), 
                                                 fill=None, edgecolor="red", linewidth=0.1)) 
        else:
            if SAVE_PLOT:
                plot.add_patch(patches.Rectangle((xr_min, yr_min), (xr_max-xr_min), (yr_max-yr_min), 
                                                 fill=None, edgecolor="green", linewidth=0.1))

        xr_center = xr_max - xr_width/2
        xr_min = xr_center - FFT_FRAME_RES/2
        xr_max = xr_center + FFT_FRAME_RES/2
        
        if (xr_min >= 0 and xr_max <= len(spectogram) and xr_center not in not_allowed_centers_list):
            new_not_allowed_centers = range(xr_center-FFT_FRAME_RES/4, xr_center+FFT_FRAME_RES/4)
            not_allowed_centers_list = not_allowed_centers_list + new_not_allowed_centers
            yr_min = 0
            yr_max = FFT_FRAME_RES

            segment_frame = [xr_min, xr_max, yr_min, yr_max]
            subpic = np.array(spectogram[xr_min:xr_max,yr_min:yr_max])

            resized_subpic = scipy.misc.imresize(subpic, (SEGMENT_SIZE, SEGMENT_SIZE), interp='nearest')
            
            SPEC_SEGMENTS.append(np.array(resized_subpic)) 

    if SAVE_PLOT:
        fig.savefig(folder+ "single_specs/"+fname+'_segments.png', bbox_inches='tight', dpi=600)
        fig.clear()
            
    return SPEC_SEGMENTS
                




def pic_to_ubyte (pic):
    a = (pic-np.min(pic) ) /(np.max(pic - np.min(pic))) 
    a = img_as_ubyte(a)
    return a

def stft(sig, frameSize, frameRes, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1

    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.fft(frames, frameRes)  

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
 
###############################
## Create the Spectrograms
###############################  
counter = 0
print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())  
for rec in recordings:
    fname = rec[1]
    pickle_path = segments_folder + fname + ".pickle"
    if (not os.path.isfile(pickle_path)):
        wav_path = "{0}{1}.wav".format(wav_folder, rec[1])
        if (os.path.isfile(wav_path)):
            signal, fs = get_wav_info("{0}{1}.wav".format(wav_folder, rec[1]))
            spectogram = abs(stft(signal, FFT_FRAME_SIZE, FFT_FRAME_RES))

            segments = get_segments(fname, spectogram)
            print fname + " " + str(np.array(segments).shape)
            with open(segments_folder + fname + ".pickle", 'wb') as f:
                pickle.dump(segments, f, protocol=-1)
    if (counter % 100 == 0):
        print "{0} - {1}".format(counter, strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())) 
    counter += 1
print strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
# print 
# print "Segments generated"
# print "Sampling rate: " + str(SAMPLING_RATE)
# print "1 frame = {0} milliseconds".format(np.floor((FFT_FRAME_SIZE/float(SAMPLING_RATE))*1000))
# print "1 second = {0} frames".format(float(SAMPLING_RATE)*2/FFT_FRAME_SIZE)
# print "1 bin = {0} Hz".format(SAMPLING_RATE/(FFT_FRAME_RES/2))
# print "1 segment = {0} seconds".format()

