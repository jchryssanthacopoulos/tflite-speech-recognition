"""Write tensorflow lite model."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import argparse

from tensorflow import lite
from tensorflow.keras import models


def write_lite(input_file, output_file):
    model = models.load_model(input_file)
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_file, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write tensorflow lite model')
    parser.add_argument('-i', '--input', type=str, required=True, help='Name of input file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Name of output file')
    arguments = parser.parse_args()

    write_lite(arguments.input, arguments.output)
