import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

# pd.options.display.max_columns = 164

emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
root = 'dataset/recordings'
duration_sec = 3


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
                        sr = 22050 * 2,
                        offset = 0.1)


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


separated_df = separate_columns(features_df)
labeled_df = pd.concat([separated_df, files_df["emotion"]], axis = 1)
labeled_df = drop_missing_rows(labeled_df)


def plot_time_series(samples):
    plt.figure(figsize = (14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(samples)), samples)
    plt.show()


def noise(samples):
    noise_amp = 0.005 * np.random.uniform() * np.amax(samples)
    samples = samples.astype('float64') + noise_amp * np.random.normal(size = samples.shape[0])

    return samples


def shift(samples):
    s_range = int(np.random.uniform(low = -5, high = 5) * 500)

    return np.roll(samples, s_range)


def stretch(samples, rate=0.8):
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


def noise_augmentation():
    data = pd.DataFrame(columns = ['feature', 'label'])

    for row in tqdm(range(len(files_df))):
        samples, sample_rate = load_wav_file(row)

        if files_df.emotion[row]:
            samples = noise(samples)
            mfccs = get_mfcc(samples, sample_rate)
            data.loc[row] = [np.mean(mfccs, axis = 0), files_df.emotion[row]]

    return data


noised_data = noise_augmentation()

noised_data = separate_columns(noised_data)
noised_data = pd.concat([noised_data, files_df["emotion"]], axis = 1)
noised_data = drop_missing_rows(noised_data)

data_df = pd.concat([labeled_df, noised_data], ignore_index = True)
data_df.to_csv("dataset/csv/data.csv", index = False)


