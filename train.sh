# Train and save the model


if [ -f .env ]; then
  source .env
else
  source .env.example
fi


# extract features into file
python extract_features.py -i $INPUT_DIR -o $FEATURES_FILE


# train model
docker build -t tflite-speech-recognition . --build-arg FEATURES_FILE=$FEATURES_FILE
chmod o+x train_model_gpu.sh
./train_model_gpu.sh $FEATURES_FILE $MODEL_FILE $DETECT_WORDS


# convert to tensorflow lite
python write_lite_model.py -i $MODEL_FILE -o $MODEL_LITE_FILE
