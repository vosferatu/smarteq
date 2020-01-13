import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys
import eq

CATEGORIES = ['Blues', 'Classical', 'Country', 'Disco',
              'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

filename = sys.argv[1]
y, sr = librosa.load(filename, mono=True, duration=30)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rmse = librosa.feature.rms(y=y)[0]
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
for e in mfcc:
    to_append += f' {np.mean(e)}'

model = load_model('model.h5')
# summarize model.
model.summary()

# evaluate loaded model on test data
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

X_new = np.fromstring(to_append, dtype=float, sep=' ')
print(X_new)


ynew = model.predict(X_new.reshape(1, -1))

print(ynew)

ynew = ynew.astype(int)
print("X=%s, Predicted=%s" % (X_new, ynew))

print(CATEGORIES[np.argmax(ynew)])

eq.equalizer(filename, CATEGORIES[np.argmax(ynew)])
