import os
from librosa import display
import matplotlib.pyplot as plt
import librosa
import numpy as np
from tqdm import tqdm

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

spectrogram_path = "dataset/spectrogram_images"
val_path = "dataset/divided_dataset/validation"
train_path = "dataset/divided_dataset/train"
test_path = "dataset/divided_dataset/test"


def create_mfccs_plots(file_dir, bath_name):
    for name in class_names:
        files_path = os.path.join(file_dir, name)

        for file in tqdm(os.listdir(files_path)):
            mfccs = get_mfccs(files_path, file)
            file_path = os.path.join(spectrogram_path, bath_name, name, file[:-4] + ".png")

            make_plot()
            save_plot(file_path, mfccs)


def get_mfccs(files_path, file):
    samples, sample_rate = librosa.load(os.path.join(files_path, file),
                                        res_type = 'kaiser_fast',
                                        duration = 3,
                                        offset = 0.1)

    mfccs = librosa.feature.mfcc(y = samples, sr = sample_rate, n_mfcc = 25)

    return mfccs


def make_plot():
    fig = plt.figure(figsize = [0.72, 0.72])

    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)


def save_plot(file_path, mfccs):
    librosa.display.specshow(librosa.power_to_db(mfccs, ref = np.max))

    plt.savefig(file_path, dpi = 400, bbox_inches = 'tight', pad_inches = 0)
    plt.close('all')


create_mfccs_plots(val_path, "validation")
create_mfccs_plots(train_path, "train")
create_mfccs_plots(test_path, "test")
