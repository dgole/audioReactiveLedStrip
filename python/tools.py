from __future__ import print_function
from __future__ import division
import time
import sys
import numpy as np
from numpy import *
from scipy.ndimage.filters import gaussian_filter1d
import config

#####################################
# Matricies to do the music manipulations
#####################################
# this is pixels x pixels, picks out certain notes based on given scale
def getPixelPixelMatrix(noteList):
    matrix = np.zeros([config.N_PIXELS, config.N_PIXELS])
    for i in range(config.N_PIXELS):
        for j in range(config.N_PIXELS):
            if (j-i%12)%12 in noteList:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = 0.0
    return matrix
# this is 12 x pixels, lets you sum up how much of the given scale is in the spectrum for each possible key
def getScalePixelMatrix(noteList):
    matrix = np.zeros([12, config.N_PIXELS])
    for i in range(12):
        for j in range(config.N_PIXELS):
            if (j-i%12)%12 in noteList:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = 0.0
    return matrix


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


class Note:
    def __init__(self, alpha, thresh):
	self.thresh=thresh
        self.sums = np.ones(12)
        self.matrix = getScalePixelMatrix([0])
        self.alpha = alpha
        self.noteStringList = ['c', 'cs', 'd', 'ef', 'e', 'f', 'fs', 'g', 'af', 'a', 'bf', 'b' ]
        self.uniqueNoteHist = ['aaaa']
    def update(self, newValues):
        newSums = np.dot(self.matrix, newValues)
        self.sums = self.alpha * newSums + (1.0 - self.alpha) * self.sums
        if (np.amax(self.sums) / np.sum(self.sums)) > self.thresh and self.noteStringList[self.sums.argmax()] != self.uniqueNoteHist[-1]:
            self.uniqueNoteHist.append(self.noteStringList[self.sums.argmax()])
    def printNoteHist(self):
        print("past notes are " + str(self.uniqueNoteHist[-10:]))
    def printCurrentNote(self):
        print("most likely note is " + self.noteStringList[self.sums.argmax()])
        print(np.amax(self.sums) / np.sum(self.sums))
        
def notePatternCheck(noteObj, notePattern):
    n = len(notePattern)
    if noteObj.uniqueNoteHist[-n:] == notePattern:
        return True
    else:
        return False
        

class Key:
    def __init__(self, matrix, alpha):
        self.keySums = np.ones(12)
        self.matrix = matrix
        self.alpha = alpha
        self.keyStringList = ['c', 'cs', 'd', 'ef', 'e', 'f', 'fs', 'g', 'af', 'a', 'bf', 'b' ]
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

class Runner:
    def __init__(self, n, speed, color, startLoc):
        self.n = n
        self.speed = speed
        self.color = color
        self.locInt = startLoc
        self.locFloat = float(self.startLoc)
        self.outArray = np.zeros(config.N_PIXELS)
        self.outZeros = np.zeros_like(self.outArray)
        if self.speed > 0:
            self.outArray[self.startLoc-self.n:self.startLoc] = 1.0
        else:
            self.outArray[self.startLoc:self.startLoc+self.n] = 1.0
    def update(self):
        self.locFloat = self.locFloat + self.speed
        if int(self.locFloat) != self.locInt:
            self.locInt = int(self.LocFloat)
            self.outArray = numpy.roll(self.outArray, np.sign(self.speed))
    def getFullOutArray(self):
        if self.color=='r':
            return np.concatenate(self.outArray, self.outZeros, self.outZeros)
        elif self.color=='g':
            return np.concatenate(self.outZeros, self.outArray, self.outZeros)
        elif self.color=='b':
            return np.concatenate(self.outZeros, self.outZeros, self.outArray)
                    
    
        
        
        
        
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
    print('sample rate: ' + str(sample_rate))
    print('first 20 feqs \n' + str(freqs[0:20]))
    print('last 20 feqs \n' + str(freqs[-20:]))
    print('center feqs in mels \n' + str(center_frequencies_mel))
    print('center feqs in mels \n' + str(center_frequencies_hz))
    print('melmat for some of the lowest notes \n')
    print(melmat[0,0:30])
    print(melmat[1,0:30])
    print(melmat[2,0:30])
    print(melmat[3,0:30])
    print(melmat[4,0:30])
    print(melmat[5,0:30])
    print(melmat[6,0:30])
    print(melmat[7,0:30])
    return melmat, (center_frequencies_mel, freqs)

