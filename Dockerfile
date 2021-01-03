FROM tensorflow/tensorflow:latest-gpu-jupyter


WORKDIR /tflite-speech-recognition

COPY all_targets_mfcc_sets.npz /tflite-speech-recognition
COPY train_model.py /tflite-speech-recognition
