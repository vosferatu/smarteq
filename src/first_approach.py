import pandas as pd
import numpy as np
from keras import layers
from keras import models
import librosa
import sys
import os
import csv

import pathlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import keras
import warnings
warnings.filterwarnings('ignore')

CATEGORIES = ['Blues', 'Classical', 'Country', 'Disco',
              'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for j in range(1, 21):
    header += f' mfcc{j}'
header += ' label'
header = header.split()


# csv file to save data
'''
file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'../genres/{g}'):
        songname = f'../genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)[0]
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
'''
data = pd.read_csv('data.csv')
data.head()
data.shape

data = data.drop(['filename'], axis=1)

# label
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

# train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = models.Sequential()
model.add(layers.Dense(256, activation='relu',
                       input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)

test_loss, test_acc = model.evaluate(X_test, y_test)

# model.save('model1.h5')

print('test_acc: ', test_acc)

# validatiion

x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]

# training

model = models.Sequential()
model.add(layers.Dense(512, activation='relu',
                       input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=30,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)

# model.save("model.h5")
print("Saved model to disk")

filename = sys.argv[1]
y, sr = librosa.load(filename, mono=True, duration=30)
rmse = librosa.feature.rms(y=y)[0]
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
for element in mfcc:
    to_append += f' {np.mean(element)}'


X_new = np.fromstring(to_append, dtype=float, sep=' ')
print(X_new)


ynew = model.predict(X_new.reshape(1, -1))

print(ynew)

ynew = ynew.astype(int)
print("X=%s, Predicted=%s" % (X_new, ynew))

print(CATEGORIES[np.argmax(ynew)])
