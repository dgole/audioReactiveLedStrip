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
import tools


samples = None
mel_y = None
mel_x = None
tools.create_mel_bank()

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

keyObj = tools.Key(determineKeyMatrix, 0.001)
chordObj = tools.Chord(0.05)
beatObj = tools.Beat(0.5)

rawFilt = ExpFilter(np.tile(0.01, config.N_PIXELS), alpha_decay=0.99, alpha_rise=0.99)
ledFilt = ExpFilter(np.tile(0.01, config.N_PIXELS), alpha_decay=0.1, alpha_rise=0.7)
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
    temp1 = rawFilt.update(y)
    temp2 = ledFilt.update(y)
    r = temp2 * 1.0
    g = temp2 * 0.0
    b = temp2 * 1.0
    '''
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
    '''
    #output = np.array([r,g,b]) * 255
    output = np.array([np.flipud(r),np.flipud(g),np.flipud(b)]) * 255
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
