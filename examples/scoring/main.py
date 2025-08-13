import os
from wenet.cli.model import load_model

language = "chinese"
model = load_model(language)


audio_dir = os.path.sep.join([os.path.dirname(os.path.abspath(__file__)), "audios"])
for audio_file in os.listdir(audio_dir):
    label = audio_file.split("_")[-1].split(".")[0]

    audio_file_path = os.path.sep.join([audio_dir, audio_file])
    score = model.transcribe_with_label(audio_file_path, label)
    print(audio_file, label, round(score, 3))
