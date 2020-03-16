from tensorflow.keras.models import load_model
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import pandas as pd

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
rec_path = "my_recordings/output.wav"
model_path = "models/Emotion_Voice_Detection_Model_1D.h5"
model = load_model(model_path)
sample_rate = 44100
duration_sec = 10
n_mfcc = 13
step = 164


def record():
    rec = sd.rec(int(sample_rate * duration_sec),
                 samplerate = sample_rate,
                 channels = 2)
    sd.wait()

    return rec


print("rec")
recording = record()
print("rec stop")

samples = np.mean(recording, axis = 1)


def save_audio(path, rec):
    write(path, sample_rate, rec)


save_audio(rec_path, recording)


def get_mfcc(features, sr):
    return librosa.feature.mfcc(y = features, sr = sr, n_mfcc = n_mfcc)


mfccs = get_mfcc(samples, sample_rate)


def split_array(array, interval):
    array_mean = np.mean(array, axis = 0)
    return [array_mean[i:i + interval] for i in range(0, len(array[0]), interval)]


def get_features_df(array):
    return pd.DataFrame(array[i] for i in range(len(array)) if len(array[i]) == step)


mfcss_parts = split_array(mfccs, step)
df = get_features_df(mfcss_parts)

X_test = np.array(df)
X_test_cnn = np.expand_dims(X_test, axis = 2)

prediction = model.predict(X_test_cnn)
prediction_max = [np.argmax(prediction[i]) for i in range(len(prediction))]

print(prediction_max)


def predict_emotion(array):
    return class_names[np.bincount(array).argmax()]


print(predict_emotion(prediction_max))
