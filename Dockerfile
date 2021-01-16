FROM tensorflow/tensorflow:latest-gpu-jupyter


ARG FEATURES_FILE

WORKDIR /tflite-speech-recognition


COPY $FEATURES_FILE /tflite-speech-recognition
COPY train_model.py /tflite-speech-recognition
