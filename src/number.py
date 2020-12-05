from keras import layers
from keras import models
import warnings
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import csv
import pathlib
from PIL import Image
import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CATEGORIES = ['Blues', 'Classical', 'Country', 'Disco',
              'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

warnings.filterwarnings('ignore')


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

data = pd.read_csv('data.csv')
data.head()
data.shape
data = data.drop(['filename'], axis=1)


genre_list = data.iloc[:, -1]
print(genre_list)
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(y_train)
len(y_test)
X_train[10]


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
print('test_acc: ', test_acc)


x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]


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


results

predictions = model.predict(X_test)

print(X_test[0])

print(predictions[0])

print(np.sum(predictions[0]))

print(np.argmax(predictions[0]))


y, sr = librosa.load('../genres/blues/blues.00091.wav',
                     mono=True, duration=30)
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

X_new = np.fromstring(to_append, dtype=float, sep=' ')
print(X_new)
print(X_test)

X_test = np.insert(X_test, 0, X_new, axis=0)

print(X_test)

prever = model.predict(X_test)
print(prever)

print(prever[0])

print(np.sum(prever[0]))

print(np.argmax(prever[0]))
print(np.sum(prever[0]))

print(CATEGORIES[np.argmax(prever[0])])
