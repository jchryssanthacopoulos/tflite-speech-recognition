"""Extract features from sound files."""

from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features


# Settings
feature_sets_file = 'all_targets_mfcc_sets.npz'
perc_keep_samples = 1.0 # 1.0 is keep all samples
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 16
len_mfcc = 16
dataset_path = 'data/data_speech_commands_v0.02'


def get_filenames_and_targets(dataset_path)
    # Create an all targets list
    all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
    all_targets.remove('_background_noise_')
    all_targets.remove('bella')

    # Create list of filenames along with ground truth vector (y)
    filenames = []
    y = []
    for index, target in enumerate(all_targets):
        print(join(dataset_path, target))
        files = listdir(join(dataset_path, target))
        print(f"Adding {len(files)} samples")
        filenames.append(files)
        y.append(np.ones(len(filenames[index])) * index)

    return filenames, y


# Flatten filename and y vectors
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]


# Associate filenames with true output and shuffle
filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)


# Only keep the specified number of samples (shorter extraction/training)
print(len(filenames))
filenames = filenames[:int(len(filenames) * perc_keep_samples)]
print(len(filenames))


# Calculate validation and test set sizes
val_set_size = int(len(filenames) * val_ratio)
test_set_size = int(len(filenames) * test_ratio)


# Break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]


# Break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]


def signal_to_mfcc(signal, fs):
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


# Function: Create MFCC from given path
def calc_mfcc(path):
    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)
    return signal_to_mfcc(signal, fs)


if __name__ == '__main__':
    """Extract features from directory of sound files."""
    parser = argparse.ArgumentParser(description='Train tensorflow CNN model')
    parser.add_argument('-i', '--input', type=str, help='File of precomputed features')
    parser.add_argument('-o', '--output', type=str, help='Trained model file')
    parser.add_argument('-w', '--wake-words', type=str, help='Comma-separated list of wake words to train to detect')
    parser.add_argument('--require-gpu', action='store_true', help='Error out if GPU unavailable')
    parser.add_argument('--save-plots', action='store_true', help='Whether to save performance plots')
    arguments = parser.parse_args()
