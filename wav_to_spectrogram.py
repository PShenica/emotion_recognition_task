import os
from librosa import display
import matplotlib.pyplot as plt
import librosa
import numpy as np

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# files dir
spectrogram_path = "Dataset/spectrogram/"
train_dir = "Dataset/train"
val_dir = "Dataset/val"


def data_to_png_format(file_dir, data_name):
    for name in class_names:
        items_path = os.path.join(file_dir, name)

        for item in os.listdir(items_path):
            samples, sample_rate = librosa.load(os.path.join(items_path, item), res_type='kaiser_fast', duration=3)
            mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=25)

            fig = plt.figure(figsize=[0.72, 0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

            filename = spectrogram_path + data_name + "/" + name + "/" + item + ".png"
            # s = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
            librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max))
            plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')


# if not os.listdir("Dataset/spectrogram/train/angry"):
data_to_png_format(val_dir, "val")
data_to_png_format(train_dir, "train")
