import os
from pydub import AudioSegment, silence
from tqdm import tqdm

dataset_path = os.path.join('dataset')
class_names = ['neutral', 'surprise', 'happy', 'angry', 'sad', 'fear']
audio_path = 'ramas'
new_audio_path = 'recordings'

duration = 2 * 1000
silence_db = -55
max_silence_duration_in_percent = 30
min_silence_len = 100


class Audio:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.audio_segment = get_audio(self.path)
        self.audio_duration = get_audio_duration(self.audio_segment)


def get_audio(path):
    return AudioSegment.from_wav(path)


def get_audio_duration(audio_segment):
    return int(audio_segment.duration_seconds * 1000)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_audio_fraction(audio, sec):
    return audio.audio_segment[sec + 1:(sec + duration)]


def get_silence_intervals(audio_fraction):
    return silence.detect_silence(
        audio_fraction,
        min_silence_len = min_silence_len,
        silence_thresh = silence_db
        )


def get_silence_duration(silence_intervals):
    return sum(map(lambda x: x[1] - x[0], silence_intervals))


def determine_silence(audio, dest_path):
    for sec in range(0, audio.audio_duration - audio.audio_duration % duration, duration):
        audio_fraction = get_audio_fraction(audio, sec)
        silence_intervals = get_silence_intervals(audio_fraction)
        silence_duration = get_silence_duration(silence_intervals)

        if silence_duration < duration * max_silence_duration_in_percent / 100:
            new_name = str(audio.name.split('.')[0]) + f'_{sec}.wav'
            audio_fraction.export(os.path.join(dest_path, new_name), format = "wav")


def handle_audio(source_path, dest_path):
    for audio_name in tqdm(os.listdir(source_path)):
        file_path = os.path.join(source_path, audio_name)
        audio = Audio(file_path, audio_name)

        if audio.audio_duration > duration:
            determine_silence(audio, dest_path)


def split_audio():
    for class_name in class_names:
        source_path = os.path.join(dataset_path, audio_path, class_name)
        dest_path = os.path.join(dataset_path, new_audio_path, class_name)

        create_folder(dest_path)
        handle_audio(source_path, dest_path)


split_audio()
