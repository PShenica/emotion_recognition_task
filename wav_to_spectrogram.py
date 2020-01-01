import os
from librosa import display
import matplotlib.pyplot as plt
import librosa
import numpy as np
import shutil
from tqdm import tqdm

# directory names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
bath_names = {"train": "train", "val": "validation"}
root_names = {"spectrogram": "spectrogram_png", "dataset": "divided_dataset"}

# files dir
dataset_path = "dataset"
spectrogram_path = os.path.join(dataset_path, root_names["spectrogram"])
recordings_path = os.path.join(dataset_path, "recordings")
val_path = os.path.join(dataset_path, root_names["dataset"], bath_names["val"])
train_path = os.path.join(dataset_path, root_names["dataset"], bath_names["train"])


def clear_files_dir():
    """
        If directory exist delete all files in dir and dir
        Then create new empty directories
    """
    for root in root_names.values():
        root_path = os.path.join(dataset_path, root)

        # check if folder already exist and delete
        if os.path.exists(root_path):
            shutil.rmtree(root_path)

        # create empty folder
        os.mkdir(root_path)

        for bath in bath_names.values():
            bath_path = os.path.join(root_path, bath)
            os.mkdir(bath_path)

            for name in class_names:
                class_path = os.path.join(bath_path, name)
                os.mkdir(class_path)


def split_dataset():
    """Split original dataset into test and validation"""
    for class_name in class_names:
        source_dir = os.path.join(recordings_path, class_name)

        # put every 6 file into validation bath
        for index, file_name in enumerate(tqdm(os.listdir(source_dir))):

            if index % 6 != 0:
                dest_dir = os.path.join(train_path, class_name)
            else:
                dest_dir = os.path.join(val_path, class_name)

            # copy file from path to another path
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


def data_to_png_format(file_dir, bath_name):
    """Convert wav file to png spectrogram"""
    for name in class_names:
        items_path = os.path.join(file_dir, name)

        for item in tqdm(os.listdir(items_path)):
            # load wav file
            samples, sample_rate = librosa.load(os.path.join(items_path, item),
                                                res_type='kaiser_fast',
                                                duration=3,
                                                offset=0.1)
            # stft = np.abs(librosa.stft(samples))

            # different spectrogram forms
            mel = librosa.feature.melspectrogram(y = samples, sr = sample_rate)
            # mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=25)
            # chroma = librosa.feature.chroma_stft(S = stft, sr = sample_rate)
            # contrast = librosa.feature.spectral_contrast(S = stft, sr = sample_rate)
            # tonnetz = librosa.feature.tonnetz(y = librosa.effects.harmonic(samples), sr = sample_rate)

            # cut info plot from graph
            fig = plt.figure(figsize=[0.72, 0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

            # save spectrogram
            file_path = spectrogram_path + "/" + bath_name + "/" + name + "/" + item + ".png"
            librosa.display.specshow(librosa.power_to_db(mel, ref=np.max))
            plt.savefig(file_path, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')


user_input = input("clear divided dataset and spectrogram folders, and split dataset? : ")

if user_input.lower() == "yes":

    clear_files_dir()
    split_dataset()

data_to_png_format(val_path, "validation")
data_to_png_format(train_path, "train")
