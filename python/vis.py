from __future__ import print_function
from __future__ import division
import time
import sys
import numpy as np
from numpy import *
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone
import led

keyOption = sys.argv[1]

#####################################
# Exponential Decay/Growth Filter Class
#####################################
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
samples = None
mel_y = None
mel_x = None
create_mel_bank()

#####################################
# Track FPS
#####################################
_time_prev = time.time() * 1000.0
_fps = ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
def frames_per_second():
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)

#####################################
# Stuff that was already here
#####################################
def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}
    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper
@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)
def interpolate(y, new_length):
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z

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
# Actually Define a bunch of these matricies
determineKeyMatrix = getScalePixelMatrix([0,2,4,7,9,11])
diatonicMatrix = getPixelPixelMatrix([0,2,4,5,7,9,11])
nonDiatonicMatrix = getPixelPixelMatrix([1,3,6,10])
pentatonicMatrix = getPixelPixelMatrix([0,2,4,7,9])
chordMatrix = getPixelPixelMatrix([0,2,4])
tonicMatrix = getPixelPixelMatrix([0])

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
keyObj = Key(determineKeyMatrix, 0.001)

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
beatObj = Beat(0.5)

raw_filt = ExpFilter(np.tile(0.01, config.N_PIXELS), alpha_decay=0.99, alpha_rise=0.99)
led_filt = ExpFilter(np.tile(0.01, config.N_PIXELS), alpha_decay=0.1, alpha_rise=0.7)
_prev_spectrum = np.tile(0.01, config.N_PIXELS)
mel_gain = ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.05, alpha_rise=0.99)
volume = ExpFilter(config.MIN_VOLUME_THRESHOLD, alpha_decay=0.02, alpha_rise=0.02)

colorThisTime = 0
def visualize_spectrum(y):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    global _prev_spectrum, colorThisTime
    #y = np.copy(interpolate(y, config.N_PIXELS))
    _prev_spectrum = np.copy(y)
    # Color channel mappings
    keyObj.update(y)
    chordObj.update(y, keyObj.getKeyNum())
    beatObj.update(y)
    temp1 = raw_filt.update(y)
    temp2 = led_filt.update(y)
    if beatObj.beatRightNow():
        colorThisTime = (colorThisTime + 1)%3
        print("BEAT!!!!")
    if colorThisTime == 0:
        r = temp2 * 1.0
        g = temp2 * 0.0
        b = temp2 * 0.0
    elif colorThisTime == 1:
        r = temp2 * 0.0
        g = temp2 * 1.0
        b = temp2 * 0.0
    if colorThisTime == 2:
        r = temp2 * 0.0
        g = temp2 * 0.0
        b = temp2 * 1.0

    output = np.array([r, g,b]) * 255
    return output

fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()

def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update, keyGuess, keyStringList
    # Normalize samples between 0 and 1
    y = audio_samples / 2.0**15
    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)
    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        led.pixels = np.tile(0, (3, config.N_PIXELS))
        led.update()
    else:
        # Transform audio input into the frequency domain
        N = len(y_data)
        #N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Pad with zeros until the next power of two
        y_data *= fft_window
        #y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
    	#YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
    	YS = np.abs(np.fft.rfft(y_data)[:N // 2])
    	XS = np.fft.rfftfreq(N, d = 1.0 / (config.MIC_RATE))
        # Construct a Mel filterbank from the FFT data
        # Scale data to values more suitable for visualization
        mel = np.dot(mel_y, YS)
        mel = mel**2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        # Map filterbank output onto LED strip
        output = visualization_effect(mel)
        led.pixels = output
        led.update()
    
    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 2.0 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))
            keyObj.printKey()
   
                

# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

visualization_effect = visualize_spectrum
"""Visualization effect to display on the LED strip"""

if __name__ == '__main__':
    # Initialize LEDs
    led.update()
    # Start listening to live audio stream
    microphone.start_stream(microphone_update)
