from keras.models import load_model
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys

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

data = pd.read_csv('data.csv')
data.head()
data.shape
# Dropping unneccesary columns
data = data.drop(['filename'], axis=1)

# Encoding the Labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

# Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

# Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# evaluate loaded model on test data
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


X_new = np.fromstring(to_append, dtype=float, sep=' ')
print(X_new)


ynew = model.predict_classes(X_new.reshape(1, -1))
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (X_new, ynew))
