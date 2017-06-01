from __future__ import print_function
from __future__ import division
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone
import dsp
import led

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""

def frames_per_second():
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)

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
    """Intelligently resizes the array by linearly interpolating the values
    Parameters
    y : np.array  Array that should be resized
    new_length : int  The length of the new interpolated array
    Returns  z : np.array  New array with length of new_length that contains the interpolated values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z

r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS), alpha_decay=0.1, alpha_rise=0.3)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS), alpha_decay=0.1, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS), alpha_decay=0.1, alpha_rise=0.3)
_prev_spectrum = np.tile(0.01, config.N_PIXELS)

histSpec = np.zeros(config.N_PIXELS)

def getPixelPixelMatrix(noteList):
    matrix = np.zeros([config.N_PIXELS, config.N_PIXELS])
    for i in range(config.N_PIXELS):
        for j in range(config.N_PIXELS):
            if (j-i%12)%12 in noteList:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = 0.0
    return matrix

def getScalePixelMatrix(noteList):
    matrix = np.zeros([12, config.N_PIXELS])
    for i in range(12):
        for j in range(config.N_PIXELS):
            if (j-i%12)%12 in noteList:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = 0.0
    return matrix

chordRefMatrix = np.array([[0,4,7], [2,5,9], [4,7,11], [5,9,11], [7,11,2], [9,0,4], [11,2,5]])
chordMatrixList = []
for i in range(12):
    chordMatrixList.append(np.zeros([7,config.N_PIXELS]))
for keyNum in range(12):
    for chordNum in range(7):
        for pixelNum in range(config.N_PIXELS):
            if (pixelNum-keyNum%12)%12 in chordRefMatrix[chordNum]:
                chordMatrixList[keyNum][chordNum, pixelNum] = 1.0
            else:
                chordMatrixList[keyNum][chordNum, pixelNum] = 0.0
print(chordMatrixList[0])

determineKeyMatrix = getScalePixelMatrix([0,2,4,7,9,11])
diatonicMatrix = getPixelPixelMatrix([0,2,4,5,7,9,11])
nonDiatonicMatrix = getPixelPixelMatrix([1,3,6,10])
pentatonicMatrix = getPixelPixelMatrix([0,2,4,7,9])
chordMatrix = getPixelPixelMatrix([0,2,4])
tonicMatrix = getPixelPixelMatrix([0])
keyGuess = 0
chordGuess = 0
chordForSure = 0
count = 0
noteStringList = ['c', 'cs', 'd', 'ef', 'e', 'f', 'f#', 'g', 'af', 'a', 'bf', 'b' ]
chordStringList = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii']
prevChords = np.zeros(10)

def visualize_spectrum(y):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    global _prev_spectrum
    global histSpec
    global count
    global keyGuess
    global keyStringList
    global chordForSure
    y = np.copy(interpolate(y, config.N_PIXELS))
    _prev_spectrum = np.copy(y)
    keySums = np.dot(determineKeyMatrix, histSpec)
    keyGuess = keySums.argmax()
    chordSums = np.dot(chordMatrixList[keyGuess], y)
    chordGuess = chordSums.argmax()
    prevChords[count] = chordGuess    
    count=(count+1)%len(prevChords)
    if np.all(chordGuess==prevChords):
        chordForSure = chordGuess     
        #print('chord is ' + str(chordStringList[chordGuess])) 
    if histSpec.max()>0.5:
        histSpec *= 0.5
    histSpec +=(y/1000)
    # Color channel mappings
    temp = r_filt.update(y)
    if chordForSure == 0:
        r = temp * 1.0
        g = temp * 1.0
        b = temp * 1.0
    elif chordForSure == 1:
        r = temp * 1.0
        g = temp * 1.0
        b = temp * 0.0  
    elif chordForSure == 2:
        r = temp * 1.0
        g = temp * 0.5
        b = temp * 0.0  
    elif chordForSure == 3:
        r = temp * 0.0
        g = temp * 1.0
        b = temp * 0.0  
    elif chordForSure == 4:
        r = temp * 0.0
        g = temp * 0.0
        b = temp * 1.0  
    elif chordForSure == 5:
        r = temp * 1.0
        g = temp * 0.0
        b = temp * 1.0  
    else:
        r = temp * 1.0
        g = temp * 0.0
        b = temp * 0.0  
    #g = b * diatonicMatrix[keyGuess]
    #r  = b * nonDiatonicMatrix[keyGuess]
    #b = b * tonicMatrix[keyGuess] 
    output = np.array([r, g,b]) * 255
    return output

mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.02, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.7, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD, alpha_decay=0.02, alpha_rise=0.02)
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
        N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
	YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
	# Construct a Mel filterbank from the FFT data
        # Scale data to values more suitable for visualization
	mel = np.dot(dsp.mel_y, YS)
	mel = mel**2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        # Map filterbank output onto LED strip
        output = visualization_effect(mel)
        led.pixels = output
        led.update()
    
    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 2.0 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))
            print(histSpec.max())
            print('key is ' + str(noteStringList[keyGuess])) 
   
                

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
