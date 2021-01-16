#!/bin/bash


FEATURES_FILE=$1
MODEL_FILE=$2
DETECT_WORDS=$3


# create new container based on image and start it
docker create --gpus all --name dummy tflite-speech-recognition
docker container start dummy

# train
docker exec dummy python train_model.py \
    -i $FEATURES_FILE \
    -o $MODEL_FILE \
    -w $DETECT_WORDS \
    --require-gpu \
    --save-plots

# copy files from created container
docker cp dummy:/tflite-speech-recognition/$MODEL_FILE .
docker cp dummy:/tflite-speech-recognition/accuracy.png .
docker cp dummy:/tflite-speech-recognition/loss.png .

# stop and remove container
docker container stop dummy
docker container rm dummy
