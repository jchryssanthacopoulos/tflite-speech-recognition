#!/bin/bash


# create new container based on image and start it
docker create --gpus all --name dummy tflite-speech-recognition
docker container start dummy

# train
docker exec dummy python train_model.py \
    -i all_targets_mfcc_sets.npz \
    -o wake_word_stop_model_marvin.h5 \
    -w marvin \
    --require-gpu \
    --save-plots

# copy files from created container
docker cp dummy:/tflite-speech-recognition/wake_word_stop_model_marvin.h5 .
docker cp dummy:/tflite-speech-recognition/accuracy.png .
docker cp dummy:/tflite-speech-recognition/loss.png .

# stop and remove container
docker container stop dummy
docker container rm dummy
