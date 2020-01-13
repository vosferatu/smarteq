import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from scipy.signal import butter, lfilter
import sys
from playsound import playsound


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = lfilter(b, a, data)
    return filtered


def equalizer_10band(data, fs, gain1=0, gain2=0, gain3=0, gain4=0, gain5=0, gain6=0, gain7=0, gain8=0, gain9=0, gain10=0):
    band1 = bandpass_filter(data, 20, 39, fs, order=2) * 10**(gain1/20)
    band2 = bandpass_filter(data, 40, 79, fs, order=3) * 10**(gain2/20)
    band3 = bandpass_filter(data, 80, 159, fs, order=3)*10**(gain3/20)
    band4 = bandpass_filter(data, 160, 299, fs, order=3) * 10**(gain4/20)
    band5 = bandpass_filter(data, 300, 599, fs, order=3) * 10**(gain5/20)
    band6 = bandpass_filter(data, 600, 1199, fs, order=3) * 10**(gain6/20)
    band7 = bandpass_filter(data, 1200, 2399, fs, order=3) * 10**(gain7/20)
    band8 = bandpass_filter(data, 2400, 4999, fs, order=3) * 10**(gain8/20)
    band9 = bandpass_filter(data, 5000, 9999, fs, order=3) * 10**(gain9/20)
    band10 = bandpass_filter(data, 10000, 11000, fs, order=3) * 10**(gain10/20)
    signal = band1 + band2 + band3 + band4 + \
        band5 + band6 + band7 + band8 + band9 + band10
    return signal


def category(cat, data, rate):
    equalized = data  # default
    # equivalent: 123   45  6   78   9-10

    if(cat == 'Hiphop'):
        equalized = equalizer_10band(
            data, rate, -80, -80, -80, -84, -84, -90, -88, -88, -84, -84)
    if(cat == 'Disco'):
        equalized = equalizer_10band(
            data, rate, -78, -78, -78, -90, -90, -86, -94, -94, -88, -88)
    if(cat == 'Classical'):
        equalized = equalizer_10band(
            data, rate, -80, -80, -80, -84, -84, -94, -94, -82, -82, -82)
    if(cat == 'Country'):
        equalized = equalizer_10band(
            data, rate, -86, -86, -86, -90, -90, -90, -86, -86, -92, -92)
    if(cat == 'Blues'):
        equalized = equalizer_10band(
            data, rate, -82, -82, -82, -88, -88, -88, -78, -78, -78, -78)
    if(cat == 'Jazz'):
        equalized = equalizer_10band(
            data, rate, -82, -82, -82, -86, -86, -94, -86, -86, -80, -80)
    if(cat == 'Metal'):
        equalized = equalizer_10band(
            data, rate, -82, -82, -82, -88, -88, -80, -84, -84, -90, -90)
    if(cat == 'Reggae'):
        equalized = equalizer_10band(
            data, rate, -80, -80, -80, -78, -78, -94, -86, -86, -94, -94)
    if(cat == 'Rock'):
        equalized = equalizer_10band(
            data, rate, -82, -82, -82, -86, -86, -92, -86, -86, -82, -82)

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
    playsound('teste.wav')
