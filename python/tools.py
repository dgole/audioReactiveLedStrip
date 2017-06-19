from __future__ import print_function
from __future__ import division
import time
import sys
import numpy as np
from numpy import *
from scipy.ndimage.filters import gaussian_filter1d
import config



class ExpFilter:
    """Simple exponential smoothing filter"""
    def __init__(self, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
        """Small rise / decay factors = more smoothing"""
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val
    def update(self, value):
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value

class Key:
    def __init__(self, matrix, alpha):
        self.keySums = np.ones(12)
        self.matrix = matrix
        self.alpha = alpha
        self.keyStringList = ['c', 'cs', 'd', 'ef', 'e', 'f', 'f#', 'g', 'af', 'a', 'bf', 'b' ]
    def update(self, newValues):
        newKeySums = np.dot(self.matrix, newValues)
        self.keySums = self.alpha * newKeySums + (1.0 - self.alpha) * self.keySums
    def getKeyNum(self):
        return self.keySums.argmax()
    def printKey(self):
        print("most likely key is " + self.keyStringList[self.getKeyNum()])
        print(self.keySums)


class Chord:
    def __init__(self, alpha):
        # define the 7 x pixels matrix for each of 12 possible keys.  
        chordRefMatrix = np.array([[0,4,7], [2,5,9], [4,7,11], [5,9,11], [7,11,2], [9,0,4], [11,2,5]])
        self.chordMatrixList = []
        for i in range(12):
            self.chordMatrixList.append(np.zeros([7,config.N_PIXELS]))
        for keyNum in range(12):
            for chordNum in range(7):
                for pixelNum in range(config.N_PIXELS):
                    if (pixelNum-keyNum%12)%12 in chordRefMatrix[chordNum]:
                        self.chordMatrixList[keyNum][chordNum, pixelNum] = 1.0
                    else:
                        self.chordMatrixList[keyNum][chordNum, pixelNum] = 0.0
            self.chordSums = np.zeros(7)
            self.alpha = alpha
            self.chordStringList = chordStringList = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii']
    def update(self, newValues, currentKey):
        newChordSums = np.dot(self.chordMatrixList[currentKey], newValues)
        self.chordSums = self.alpha * newChordSums + (1.0 - self.alpha) * self.chordSums
    def getChordNum(self):
        return self.chordSums.argmax()
    def printChord(self):
        print("most likely chord is " + self.chordStringList[self.getChordNum()])
        #print(self.chordSums)

class Beat:
    def __init__(self, alpha):
        self.alpha = alpha
        self.matrix = np.zeros(config.N_PIXELS)
        self.sum = 0.0
        self.oldSum = 0.0
        for i in range(config.N_PIXELS):
            if i<9:
                self.matrix[i] = 1.0
            else:
                self.matrix[i] = 0.0                   
    def update(self, newValues):
        self.oldSum = self.sum
        newSum = np.dot(self.matrix, newValues)
        self.sum = self.alpha * newSum + (1.0 - self.alpha) * self.sum
    def beatRightNow(self):
        if self.sum > 2.0*self.oldSum:
            return True 
            print(self.sum, self.oldSum)
        else:
            return False


#####################################
# Create mel bank to convert frequencies to notes
#####################################
def hertz_to_mel(freq):
    return round(12.0*(np.log(0.0323963*freq)/0.693147)+12.0)
def mel_to_hertz(mel):
    return 440.0 * (2.0**(1.0/12.0))**(mel-58.0)
def melfrequencies_mel_filterbank(num_bands, freq_min, freq_max, num_fft_bands):
    mel_max = hertz_to_mel(freq_max)
    mel_min = hertz_to_mel(freq_min)
    print(freq_min, freq_max)
    print(mel_min, mel_max)
    delta_mel = abs(mel_max - mel_min) / (num_bands-1)
    delta_hz = linspace(0.0, 44100 / 2.0, num_fft_bands)[1]-linspace(0.0, 44100 / 2.0, num_fft_bands)[0]
    frequencies_mel = mel_min + delta_mel * arange(-1, num_bands + 1)
    center_frequencies_mel = frequencies_mel[1:-1]
    lower_edges_mel = zeros_like(center_frequencies_mel)
    upper_edges_mel = zeros_like(center_frequencies_mel)
    for i in range(len(center_frequencies_mel)):
		cent = center_frequencies_mel[i]
		delCent =  mel_to_hertz(cent+1) - mel_to_hertz(cent)
		if delta_hz > delCent:
		    lower_edges_mel[i] = cent - 2.0  
		    upper_edges_mel[i] = cent + 2.0
		elif delta_hz < 0.5 * delCent:
		    lower_edges_mel[i] = cent - 0.5  
		    upper_edges_mel[i] = cent + 0.5
		else:
		    lower_edges_mel[i] = cent - 1.0  
		    upper_edges_mel[i] = cent + 1.0		     
    return center_frequencies_mel, lower_edges_mel, upper_edges_mel
def compute_melmat(num_mel_bands, freq_min, freq_max, num_fft_bands, sample_rate):
    center_frequencies_mel, lower_edges_mel, upper_edges_mel =  \
        melfrequencies_mel_filterbank(
            num_mel_bands,
            freq_min,
            freq_max,
            num_fft_bands
        )
    center_frequencies_hz = mel_to_hertz(center_frequencies_mel)
    lower_edges_hz = mel_to_hertz(lower_edges_mel)
    upper_edges_hz = mel_to_hertz(upper_edges_mel)
    freqs = linspace(0.0, sample_rate / 2.0, num_fft_bands)
    melmat = zeros((num_mel_bands, num_fft_bands))
    for imelband, (center, lower, upper) in enumerate(zip(center_frequencies_hz, lower_edges_hz, upper_edges_hz)):
        left_slope = (freqs >= lower) == (freqs <= center)
        melmat[imelband, left_slope] = ((freqs[left_slope] - lower) / (center - lower))
        right_slope = (freqs >= center) == (freqs <= upper)
        melmat[imelband, right_slope] = ((upper - freqs[right_slope]) / (upper - center))
    print(sample_rate)
    print(freqs[0:20])
    print(center_frequencies_mel)
    print(center_frequencies_hz)
    print(melmat[7,0:30])
    print(melmat[8,0:30])
    print(melmat[9,0:30])
    print(melmat[10,0:30])
    return melmat, (center_frequencies_mel, freqs)
def create_mel_bank():
    global samples, mel_y, mel_x
    samples = int(config.MIC_RATE * config.N_ROLLING_HISTORY / (2.0 * config.FPS))
    mel_y, (_, mel_x) = compute_melmat(num_mel_bands=config.N_FFT_BINS,
                                             freq_min=config.MIN_FREQUENCY,
                                               freq_max=config.MAX_FREQUENCY,
                                               num_fft_bands=samples,
                                               sample_rate=config.MIC_RATE)