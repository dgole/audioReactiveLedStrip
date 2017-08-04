"""Settings for audio reactive LED strip"""
from __future__ import print_function
from __future__ import division
import os

DEVICE = 'pi'

if DEVICE == 'pi':
    LED_PIN = 18
    """GPIO pin connected to the LED strip pixels (must support PWM)"""
    LED_FREQ_HZ = 800000
    """LED signal frequency in Hz (usually 800kHz)"""
    LED_DMA = 5
    """DMA channel used for generating PWM signal (try 5)"""
    BRIGHTNESS = 200
    """Brightness of LED strip between 0 and 255"""
    LED_INVERT = False
    """Set True if using an inverting logic level converter"""
    SOFTWARE_GAMMA_CORRECTION = True
    """Set to True because Raspberry Pi doesn't use hardware dithering"""

USE_GUI = False
DISPLAY_FPS = True
N_PIXELS = 120
GAMMA_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'gamma_table.npy')
MIC_RATE = 44100
"""Sampling frequency of the microphone in Hz"""
FPS = 60
_max_led_FPS = int(((N_PIXELS * 30e-6) + 50e-6)**-1.0)
assert FPS <= _max_led_FPS, 'FPS must be <= {}'.format(_max_led_FPS)

# 130.81 is c3
MIN_FREQUENCY = 130.81 * 1.0
"""Frequencies below this value will be removed during audio processing"""

# 4066.84 is b7 and a half
# 3951.066 is b7
MAX_FREQUENCY = 3951.066 * 1.0
"""Frequencies above this value will be removed during audio processing"""

N_FFT_BINS = N_PIXELS
"""Number of frequency bins to use when transforming audio to frequency domain

Fast Fourier transforms are used to transform time-domain audio data to the
frequency domain. The frequencies present in the audio signal are assigned
to their respective frequency bins. This value indicates the number of
frequency bins to use.

A small number of bins reduces the frequency resolution of the visualization
but improves amplitude resolution. The opposite is true when using a large
number of bins. More bins is not always better!

There is no point using more bins than there are pixels on the LED strip.
"""

N_ROLLING_HISTORY = 2
"""Number of past audio frames to include in the rolling window"""

MIN_VOLUME_THRESHOLD = 5e-3
"""No music visualization displayed if recorded audio volume below threshold"""
