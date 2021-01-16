"""Connect a resistor and LED to each indicated pin and light them using commands."""

import argparse
import glob

import numpy as np
import python_speech_features
import RPi.GPIO as GPIO
import sounddevice as sd
from scipy.io import wavfile
import scipy.signal
import timeit
from tflite_runtime.interpreter import Interpreter


# Parameters
rec_duration = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16


# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# GPIO 
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)


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


# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status, interpreter, input_details, output_details, save_stream, debug_mode):
    global state
    global detect_words
    global detect_pins
    global word_threshold

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    if save_stream:
        recording_files = glob.glob('recording_*.wav')
        cnt = 1
        if recording_files:
            cnt = max([int(f.strip('recording_').strip('.wav')) for f in recording_files]) + 1
        wavfile.write(f"recording_{cnt:04}.wav", resample_rate, window)

    # Compute features
    mfccs = python_speech_features.base.mfcc(window, 
                                        samplerate=new_fs,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=num_mfcc,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)
    mfccs = mfccs.transpose()

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # take all elements after the first (first element = "all other words")
    val = output_data[0][1:]

    max_idx = val.argmax()
    if val[max_idx] >= word_threshold:
        detected_word = detect_words[max_idx]
        print(f"Detected: {detected_word}")

        if state != detected_word:
            state = detected_word
            for idx, pin in enumerate(detect_pins):
                if idx == max_idx:
                    GPIO.output(pin, GPIO.HIGH)
                else:
                    GPIO.output(pin, GPIO.LOW)

    if debug_mode:
        print('Prediction:', val)
        print('Inference time:', timeit.default_timer() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect stop word')
    parser.add_argument('-i', '--input', type=str, required=True, help='Name of model file')
    parser.add_argument('-w', '--words', type=str, required=True, help='Comma-separated list of words to detect')
    parser.add_argument('-p', '--pins', type=str, required=True, help='Comma-separated list of GPIO pins to show detected words')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='Threshold above which to consider word detected')
    parser.add_argument('--save-stream', action='store_true', help='Save streaming audio streams as wav files')
    parser.add_argument('--debug', action='store_true', help='Whether to print debug information')
    arguments = parser.parse_args()

    # start in state corresponding to first word
    detect_words = arguments.words.split(",")
    detect_pins = [int(pin) for pin in arguments.pins.split(",")]

    word_threshold = arguments.threshold

    # set first pin high and others low
    GPIO.setup(detect_pins[0], GPIO.OUT, initial=GPIO.HIGH)
    for pin in detect_pins[1:]:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    # set state to first word
    state = detect_words[0]

    # load model
    interpreter = Interpreter(arguments.input)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def sd_callback_with_args(rec, frames, time, status):
        # pass in model and arguments
        return sd_callback(
            rec, frames, time, status, interpreter, input_details, output_details,
            arguments.save_stream, arguments.debug
        )

    # Start streaming from microphone
    print("Recording...")
    with sd.InputStream(channels=num_channels,
                        samplerate=sample_rate,
                        blocksize=int(sample_rate * rec_duration),
                        callback=sd_callback_with_args):
        while True:
            pass
