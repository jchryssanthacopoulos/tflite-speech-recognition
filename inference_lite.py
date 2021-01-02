"""Run inference with tensorflow lite modelon directory of sound files."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import glob
import sys

import numpy as np
import python_speech_features
from scipy.io import wavfile
import scipy.signal
from tflite_runtime.interpreter import Interpreter


resample_rate = 8000
num_mfcc = 16


# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs


def calc_mfcc(path):
    # Load wavefile
    sample_rate, signal = wavfile.read(path)
    if sample_rate != resample_rate:
        signal, sample_rate = decimate(signal, sample_rate, resample_rate)

    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal, 
                                            samplerate=sample_rate,
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


def get_model_score(model, input_details, output_details, sound_file):
    mfccs = calc_mfcc(sound_file)
    mfccs = mfccs.reshape((1, num_mfcc, num_mfcc, 1))

    model.set_tensor(input_details[0]['index'], np.float32(mfccs))
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])

    return output_data[0][0]


def load_lite_model(model_name):
    interpreter = Interpreter(model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


if __name__ == '__main__':
    # Usage is: python inference_lite.py <model_path> <sound_files_path>
    model_name, directory = sys.argv[1], sys.argv[2]

    model, input_details, output_details = load_lite_model(model_name)
    sound_files = sorted(glob.glob(f"{directory}/*.wav"))

    for sound_file in sound_files:
        score = get_model_score(model, input_details, output_details, sound_file)
        print(f"Score on file {sound_file}: {score}")
