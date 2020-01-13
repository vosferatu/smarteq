import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from scipy.signal import butter, lfilter
import sys
from playsound import playsound

lowcut_frequencies = [20, 40, 80, 160, 300, 600, 1200, 2400, 5000, 10000]
highcut_frequencies = [39, 79, 159, 299, 599, 1199, 2399, 4999, 9999, 10999]

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = lfilter(b, a, data)
    return filtered


def equalizer_10band(data, fs, gains):
    signal = []
    for i in range(len(gains)):
        signal.append(bandpass_filter(data, lowcut_frequencies[i], highcut_frequencies[i], fs, order=3) * 10**(gains[i]/20))

    return sum(signal)


def category(cat, data, rate):
    equalized = data  # default
    # equivalent: 123   45  6   78   9-10

    if(cat == 'Hiphop'):
        equalized = equalizer_10band(
            data, rate, [-80, -80, -80, -84, -84, -90, -88, -88, -84, -84])
    elif(cat == 'Disco'):
        equalized = equalizer_10band(
            data, rate, [-78, -78, -78, -90, -90, -86, -94, -94, -88, -88])
    elif(cat == 'Classical'):
        equalized = equalizer_10band(
            data, rate, [-80, -80, -80, -84, -84, -94, -94, -82, -82, -82])
    elif(cat == 'Country'):
        equalized = equalizer_10band(
            data, rate, [-86, -86, -86, -90, -90, -90, -86, -86, -92, -92])
    elif(cat == 'Blues'):
        equalized = equalizer_10band(
            data, rate, [-82, -82, -82, -88, -88, -88, -78, -78, -78, -78])
    elif(cat == 'Jazz'):
        equalized = equalizer_10band(
            data, rate, [-82, -82, -82, -86, -86, -94, -86, -86, -80, -80])
    elif(cat == 'Metal'):
        equalized = equalizer_10band(
            data, rate, [-82, -82, -82, -88, -88, -80, -84, -84, -90, -90])
    elif(cat == 'Reggae'):
        equalized = equalizer_10band(
            data, rate, [-80, -80, -80, -78, -78, -94, -86, -86, -94, -94])
    elif(cat == 'Rock'):
        equalized = equalizer_10band(
            data, rate, [-82, -82, -82, -86, -86, -92, -86, -86, -82, -82])
    elif(cat == 'Pop'):
        equalized = equalizer_10band(
            data, rate, [-92, -92, -92, -86, -86, -80, -88, -88, -94, -94])

    return equalized


def equalizer(file, cat):
    rate, data = wav.read(file)

    N = len(data)
    t = 1/rate * np.arange(N)
    f = rate/N * np.arange(N)

    # original signal's fft
    F_data = np.fft.fft(data)/N

    # applying equalizer
    new_sound = category(cat, data, rate)

    wav.write("teste.wav", rate, new_sound)
    print('Playing...')
    playsound('teste.wav')
