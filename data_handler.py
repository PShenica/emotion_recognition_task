import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
root = 'dataset/recordings'
duration_sec = 2
offset = 0.1
sr = 22050 * 2


def get_files_df():
    data = []

    for emotion in emotion_names:
        files_path = os.path.join(root, emotion)

        for file in os.listdir(files_path):
            path = os.path.join(files_path, file)
            data.append([path, emotion])

    return pd.DataFrame(data, columns = ["path", "emotion"])


files_df = get_files_df()


def load_wav_file(row):
    return librosa.load(files_df.path[row],
                        res_type = 'kaiser_fast',
                        duration = duration_sec,
                        sr = sr,
                        offset = offset)


def get_mfcc(samples, sample_rate):
    return librosa.feature.mfcc(y = samples, sr = sample_rate, n_mfcc = 13)


def get_features_df():
    df = pd.DataFrame(columns=['feature'])

    for row in tqdm(range(len(files_df))):
        samples, sample_rate = load_wav_file(row)
        mfccs = get_mfcc(samples, sample_rate)
        df.loc[row] = [np.mean(mfccs, axis = 0)]

    return df


features_df = get_features_df()


def separate_columns(df):
    return pd.DataFrame(df['feature'].values.tolist())


def drop_missing_rows(df):
    return df.dropna()


def prepare_data(df):
    df = separate_columns(df)
    df = pd.concat([df, files_df["emotion"]], axis = 1)
    df = drop_missing_rows(df)

    return df


labeled_df = prepare_data(features_df)


def plot_time_series(samples):
    plt.figure(figsize = (14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(samples)), samples)
    plt.show()


def noise(samples, *args):
    noise_amp = 0.005 * np.random.uniform() * np.amax(samples)
    samples = samples.astype('float64') + noise_amp * np.random.normal(size = samples.shape[0])

    return samples


def shift(samples, *args):
    s_range = int(np.random.uniform(low = -5, high = 5) * 500)

    return np.roll(samples, s_range)


def stretch(samples, *args, rate=0.8):
    samples = librosa.effects.time_stretch(samples, rate)

    return samples


def pitch(samples, sample_rate):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    samples = librosa.effects.pitch_shift(samples.astype('float64'),
                                          sample_rate,
                                          n_steps = pitch_change,
                                          bins_per_octave = bins_per_octave)

    return samples


def random_value_change(samples, *args):
    change = np.random.uniform(low = 1.5, high = 3)

    return samples * change


def make_augmentation(func):
    data = pd.DataFrame(columns = ['feature', 'label'])

    for row in tqdm(range(len(files_df))):
        samples, sample_rate = load_wav_file(row)

        samples = func(samples, sample_rate)
        mfccs = get_mfcc(samples, sample_rate)
        data.loc[row] = [np.mean(mfccs, axis = 0), files_df.emotion[row]]

    return data


aug_data = []

noised_data = make_augmentation(noise)
noised_data = prepare_data(noised_data)
aug_data.append(noised_data)

pitched_data = make_augmentation(pitch)
pitched_data = prepare_data(pitched_data)
aug_data.append(pitched_data)

data_df = pd.concat(aug_data, ignore_index = True)

data_df.to_csv("dataset/csv/data.csv", index = False)
