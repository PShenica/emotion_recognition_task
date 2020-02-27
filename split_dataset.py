import os
import shutil
from tqdm import tqdm

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
bath_names = {"train": "train", "val": "validation", "test": "test"}
root_names = {"spectrogram": "spectrogram_images", "dataset": "divided_dataset"}

dataset_path = "dataset"
recordings_path = os.path.join(dataset_path, "recordings")
val_path = os.path.join(dataset_path, root_names["dataset"], bath_names["val"])
train_path = os.path.join(dataset_path, root_names["dataset"], bath_names["train"])
test_path = os.path.join(dataset_path, root_names["dataset"], bath_names["test"])


def split_dataset():
    for class_name in class_names:
        source_dir = os.path.join(recordings_path, class_name)

        copy_files_from_dir(source_dir, class_name)


def copy_files_from_dir(source_dir, class_name):
    for index, file_name in enumerate(tqdm(os.listdir(source_dir))):
        dest_dir = get_dest_path(index, class_name)

        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


def get_dest_path(index, class_name):
    if index % 6 == 0:
        path = os.path.join(val_path, class_name)
    elif index % 7 == 0:
        path = os.path.join(test_path, class_name)
    else:
        path = os.path.join(train_path, class_name)

    return path


split_dataset()
