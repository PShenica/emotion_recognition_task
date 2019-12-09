import os
import pandas as pd
import librosa
import numpy as np
import shutil
from tqdm import tqdm
from sklearn.utils import shuffle


class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# files dir
data_root = "Dataset/recordings"
train_dir = "Dataset/train"
val_dir = "Dataset/val"

# check if a directory is empty
if not os.listdir("Dataset/train/angry"):
    # split dataset into test and validation
    for class_name in class_names:
        source_dir = os.path.join(data_root, class_name)

        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % 6 != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)

            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


# data preparation
def data_to_csv_format(file_dir, csv_file_name):
    feeling_list = []

    # find out emotion
    for name in class_names:
        for _ in os.listdir(os.path.join(file_dir, name)):
            feeling_list.append(name)

    labels = pd.DataFrame(feeling_list)

    df = pd.DataFrame(columns=['feature'])
    index = 0

    # getting the features of audio files using librosa
    for name in class_names:
        items_path = os.path.join(file_dir, name)

        for item in os.listdir(items_path):
            y, sr = librosa.load(os.path.join(items_path, item), res_type='kaiser_fast', duration=2)
            sr = np.array(sr)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25), axis=0)

            feature = mfccs
            df.loc[index] = [-(feature/100)]
            index += 1

    df = pd.DataFrame(df['feature'].values.tolist())
    df_with_labels = pd.concat([df, labels], axis=1)
    df_with_labels = df_with_labels.rename(index=str, columns={"0": "label"})

    df_without_nan_vales = pd.DataFrame(df_with_labels.fillna(0))
    shuffled_df = shuffle(df_without_nan_vales)
    shuffled_df.to_csv(os.path.join('Dataset', csv_file_name + ".csv"))


data_to_csv_format(train_dir, 'train')
data_to_csv_format(val_dir, 'validation')
