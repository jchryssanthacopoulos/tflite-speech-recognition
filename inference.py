"""Run inference on directory of sound files."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import glob
import sys

import librosa
import numpy as np
import python_speech_features
from tensorflow.keras import layers, models


sample_rate = 8000
num_mfcc = 16


def calc_mfcc(path):
    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)

    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal, 
                                            samplerate=fs,
                                            winlen=0.256,
                                            winstep=0.050,
                                            numcep=num_mfcc,
                                            nfilt=26,
                                            nfft=2048,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning)
    return mfccs.transpose()



def get_model_score(model, sound_file):
    mfccs = calc_mfcc(sound_file)
    mfccs = mfccs.reshape((1, num_mfcc, num_mfcc, 1))
    return model.predict(mfccs)[0][0]



if __name__ == '__main__':
    # Usage is: python inference.py <model_path> <sound_files_path>
    model_name, directory = sys.argv[1], sys.argv[2]

    model = models.load_model(model_name)
    sound_files = glob.glob(f"{directory}/*.wav")

    for sound_file in sound_files:
        score = get_model_score(model, sound_file)
        print(f"Score on file {sound_file}: {score}")
