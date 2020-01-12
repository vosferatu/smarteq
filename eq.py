import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = lfilter(b, a, data)
    return filtered

def equalizer_10band (data, fs, gain1=0, gain2=0, gain3=0, gain4=0, gain5=0, gain6=0, gain7=0, gain8=0, gain9=0, gain10=0):
    band1 = bandpass_filter(data, 20, 39, fs, order=2)* 10**(gain1/20)
    band2 = bandpass_filter(data, 40, 79, fs, order=3)*10**(gain2/20)
    band3 = bandpass_filter(data, 80, 159, fs, order=3)*10**(gain3/20)
    band4 = bandpass_filter(data, 160, 299, fs, order=3)* 10**(gain4/20)
    band5 = bandpass_filter(data, 300, 599, fs, order=3)* 10**(gain5/20)
    band6 = bandpass_filter(data, 600, 1199, fs, order=3)* 10**(gain6/20)
    band7 = bandpass_filter(data, 1200, 2399, fs, order=3)* 10**(gain7/20)
    band8 = bandpass_filter(data, 2400, 4999, fs, order=3)* 10**(gain8/20)
    band9 = bandpass_filter(data, 5000, 9999, fs, order=3)* 10**(gain9/20)
    band10 = bandpass_filter(data, 5000, 9999, fs, order=3)* 10**(gain10/20)
    # band10 = bandpass_filter(data, 10000, 20000, fs, order=3)* 10**(gain10/20)
    signal = band1 + band2 + band3 + band4 + band5 + band6 + band7 + band8 + band9 + band10
    return signal


rate, data = wav.read("audio.wav")

N = len(data)
t  = 1/rate * np.arange(N) 
f  = rate/N * np.arange(N)

#computing fft of original signal
F_data = np.fft.fft(data)/N

#appying equalizer
equalized = equalizer_10band(data, rate, -100,-100,-100,0,0,0,0,0,0,0)

wav.write("teste.wav", rate, data)


#computing fft of filtered signal
# Y = np.fft.fft(equalized)/N

# plt.figure(figsize=(10, 8))
# plt.subplot(2,1,1)
# plt.plot(t, equalized,'-b',label=r"$Filtered amplitude(t)$")
# plt.xlabel('time[s]')
# plt.subplot(2,1,1)
# plt.plot(t, data,'-r',label=r"$Original amplitude(t)$")
# plt.xlabel('time[s]')
# plt.legend()
# plt.grid()
# plt.show()

# plt.subplot(2,1,2)
# plt.plot(f[:N//2],np.abs(F_data[:N//2]),'-r',label=r"$Original magnitude(f)$")
# plt.xlabel('f [Hz]')
# plt.xlim([0,5e3])
# plt.plot(f[:N//2],np.abs(Y[:N//2]),'-b',label=r"$Filtered magnitude(f)$")
# plt.xlabel('f [Hz]')
# plt.xlim([0,5e3])
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.show()