from tensorflow.keras.models import load_model
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import pandas as pd

model_path = "models/Emotion_Voice_Detection_Model_1D.h5"
model = load_model(model_path)
model_sample_shape = (164, 1)
sample_rate = 44100
n_mfcc = 13


def get_mfccs(features):
    return librosa.feature.mfcc(y = features, sr = sample_rate, n_mfcc = n_mfcc)


def split_array(array, size):
    return [array[i:i + size] for i in range(0, len(array), size)]


def make_df(array, size):
    return pd.DataFrame(array[i] for i in range(len(array)) if len(array[i]) == size)


def prepare_data(mfccs):
    array_mean = np.mean(mfccs, axis = 0)
    size = model_sample_shape[0]
    new_array = split_array(array_mean, size)
    df = make_df(new_array, size)

    return df


def predict(sample_array):
    sample_array = np.mean(sample_array, axis = 1)
    mfccs = get_mfccs(sample_array)
    df = prepare_data(mfccs)

    X_test = np.array(df)
    X_test_cnn = np.expand_dims(X_test, axis = 2)
    prediction = model.predict(X_test_cnn)

    return np.mean(prediction, axis = 0)


def record():
    rec = sd.rec(int(sample_rate * 6),
                 samplerate = sample_rate,
                 channels = 2,
                 blocking = True)

    return rec


recording = record()

print(predict(recording))
