from __future__ import print_function
from __future__ import division
import time
import sys
import numpy as np
from numpy import *
from scipy.ndimage.filters import gaussian_filter1d
import config

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


