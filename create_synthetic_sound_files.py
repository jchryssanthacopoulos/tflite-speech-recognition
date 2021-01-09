"""Create synthetic sound files using text-to-speech (TTS)."""

import argparse
import importlib
import os

from pydub import AudioSegment
import pyttsx3


class TextToSpeech:
    engine = None
    rate = None

    def __init__(self, voice_id, rate):
        importlib.reload(pyttsx3)
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', voice_id)
        self.engine.setProperty('rate', rate)

    def save_to_file(self, text, filename):
        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()


def save_text_to_file(text, filename, voice_id, rate):
    tts = TextToSpeech(voice_id, rate)
    tmp_mp3_file = "tmp_mp3_file.mp3"
    tts.save_to_file(text, tmp_mp3_file)

    sound = AudioSegment.from_mp3(tmp_mp3_file)
    sound.export(filename, format="wav")
    os.remove(tmp_mp3_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create synthetic sound files')
    parser.add_argument('-i', '--input', type=str, required=True, help='Text to generate sound files of')
    parser.add_argument('-o', '--output', type=str, required=True, help='Name of file to save')
    parser.add_argument('-n', '--num-files', type=int, default=1, help='Number of files to save per voice')
    arguments = parser.parse_args()

    engine = pyttsx3.init()
    filename_no_ext, filename_ext = os.path.splitext(arguments.output)
    # not good to use: 'aragonese', 'catalan', 'spanish-latin-am', 'spanish'
    for voice in ["en+f1", "en+m1"]: # engine.getProperty('voices'):
        voice_id = voice # voice.id
        print(f"Playing with voice {voice_id}")
        for i in range(arguments.num_files):
            for rate in [100, 150, 200]:
                recording_filename = f"{filename_no_ext}.{voice_id}.{rate}.{i + 1}{filename_ext}"
                save_text_to_file(arguments.input, recording_filename, voice_id, rate)
