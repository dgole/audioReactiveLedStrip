from __future__ import print_function
from __future__ import division
import time
import sys
import numpy as np
from numpy import *
from scipy.ndimage.filters import gaussian_filter1d
import config

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
chordObj = Chord(0.05)

