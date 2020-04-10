from tensorflow.keras.models import load_model
import numpy as np
import librosa
import pandas as pd
import io

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
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


def predict(string_bytes):
    stream = io.BytesIO(string_bytes)
    sample_array, sr = librosa.load(stream, sr = sample_rate)

    if sample_array[0] != 0:
        sample_array = np.mean(sample_array, axis = 1)

    mfccs = get_mfccs(sample_array)
    df = prepare_data(mfccs)

    X_test = np.array(df)
    X_test_cnn = np.expand_dims(X_test, axis = 2)
    prediction = model.predict(X_test_cnn)

    prediction_mean = np.mean(prediction, axis = 0)

    return class_names[np.argmax(prediction_mean)]
