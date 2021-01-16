"""Extract features from sound files."""

import argparse
from os import listdir
from os.path import isdir, join
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features


def get_filenames_and_targets(dataset_path):
    # Create an all targets list
    vocab_words = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
    vocab_words.remove('_background_noise_')
    vocab_words.remove('bella')

    # Create list of filenames along with ground truth vector (y)
    filenames = []
    y = []
    for index, word in enumerate(vocab_words):
        print(join(dataset_path, word))
        files = [join(dataset_path, word, file) for file in files]
        print(f"Adding {len(files)} samples")
        filenames.append(files)
        y.append(np.ones(len(filenames[index])) * index)

    # Flatten filename and y vectors
    filenames = [item for sublist in filenames for item in sublist]
    y = [item for sublist in y for item in sublist]

    # Associate filenames with true output and shuffle
    filenames_y = list(zip(filenames, y))
    random.shuffle(filenames_y)
    filenames, y = zip(*filenames_y)

    return vocab_words, filenames, y


# Function: Create MFCC from given path
def calc_mfcc(filename, num_mfcc, sample_rate):
    # Load wavefile
    signal, fs = librosa.load(filename, sr=sample_rate)
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


# Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y, num_mfcc, len_mfcc, sample_rate):
    prob_cnt = 0
    out_x = []
    out_y = []

    for index, filename in enumerate(in_files):
        # Check to make sure we're reading a .wav file
        if not filename.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc(filename, num_mfcc, sample_rate)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1

    return out_x, out_y, prob_cnt


def split_into_train_val_test(filenames, y, val_ratio, test_ratio):
    # Calculate validation and test set sizes
    val_set_size = int(len(filenames) * val_ratio)
    test_set_size = int(len(filenames) * test_ratio)

    # Break dataset apart into train, validation, and test sets
    filenames_split = {}
    filenames_split['val'] = filenames[:val_set_size]
    filenames_split['test'] = filenames[val_set_size:(val_set_size + test_set_size)]
    filenames_split['train'] = filenames[(val_set_size + test_set_size):]

    # Break y apart into train, validation, and test sets
    y_split = {}
    y_split['val'] = y[:val_set_size]
    y_split['test'] = y[val_set_size:(val_set_size + test_set_size)]
    y_split['train'] = y[(val_set_size + test_set_size):]

    return filenames_split, y_split


if __name__ == '__main__':
    """Extract features from directory of sound files."""
    parser = argparse.ArgumentParser(description='Extract audio features from sound files')
    parser.add_argument('-i', '--input', type=str, required=True, help='Directory of sound files')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file')
    parser.add_argument('-v', '--val-size', type=float, default=0.1, help='Fraction for validation')
    parser.add_argument('-t', '--test-size', type=float, default=0.1, help='Fraction for test')
    parser.add_argument('-n', '--num-mfccs', type=int, default=16, help='Number of MFCCs to create')
    parser.add_argument('-d', '--dim-mfcc', type=int, default=16, help='Dimension of each MFCC')
    parser.add_argument('-s', '--sample-rate', type=int, default=8000, help='Sample rate to load sound files at')
    arguments = parser.parse_args()

    # get filenames and associated word targets
    vocab_words, filenames, outputs = get_filenames_and_targets(arguments.input)

    x, y, prob = extract_features(
        filenames, outputs, arguments.num_mfcc, arguments.dim_mfcc, arguments.sample_rate
    )
    print('Removed percentage:', prob / len(outputs))

    # split into train, validation, and test
    x, y = split_into_train_val_test(x, y, arguments.val_size, arguments.test_size)

    # Save features and truth vector (y) sets to disk
    np.savez(
        arguments.output,
        x_train=x['train'],
        y_train=y['train'],
        x_val=x['val'],
        y_val=y['val'],
        x_test=x['test'],
        y_test=y['test'],
        target_words=vocab_words
    )
