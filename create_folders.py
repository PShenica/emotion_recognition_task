import os
import shutil

dataset_path = "dataset"

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
bath_names = ["train", "validation", "test"]
root_names = ["spectrogram_images", "divided_dataset"]


def make_paths(roots, names):
    paths = []

    for root in roots:
        for name in names:
            paths.append(os.path.join(root, name))

    return paths


root_paths = make_paths([dataset_path], root_names)
bath_paths = make_paths(root_paths, bath_names)
class_paths = make_paths(bath_paths, class_names)


def create_folders(paths):
    for path in paths:
        create_folder(path)


def create_folder(path):
    os.mkdir(path)


def delete_folders(paths):
    for path in paths:
        if os.path.exists(path):
            delete_folder(path)


def delete_folder(path):
    shutil.rmtree(path)


delete_folders(root_paths)
create_folders(root_paths)
create_folders(bath_paths)
create_folders(class_paths)
