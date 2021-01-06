#!/bin/bash


# create new container based on image and start it
docker create --gpus all --name dummy tflite-speech-recognition
docker container start dummy

# train
docker exec dummy python train_model.py \
    -i all_targets_mfcc_sets.npz \
    -o wake_word_stop_model_marvin.h5 \
    -w marvin \
    --require-gpu

# copy file from created container (doesn't have to be running)
docker cp dummy:/tflite-speech-recognition/wake_word_stop_model_marvin.h5 .

# stop and remove container
docker container stop dummy
docker container rm dummy
