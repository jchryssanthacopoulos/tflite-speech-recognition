"""Train CNN model for speech recognition of single word."""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(sample_shape):
    # Based on: https://www.geeksforgeeks.org/python-image-classification-using-keras/
    model = models.Sequential()
    model.add(layers.Conv2D(32,
                            (2, 2),
                            activation='relu',
                            input_shape=sample_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Display model
    model.summary()

    # Add training parameters to model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model


def transform_inputs(feature_sets):
    x_train = feature_sets['x_train']
    x_val = feature_sets['x_val']
    x_test = feature_sets['x_test']

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # CNN for TF expects (batch, height, width, channels)
    # So we reshape the input tensors with a "color" channel of 1
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    return x_train, x_val, x_test


def transform_outputs(feature_sets, wake_word):
    # Assign feature sets
    y_train = feature_sets['y_train']
    y_val = feature_sets['y_val']
    y_test = feature_sets['y_test']

    # Convert ground truth arrays to one wake word (1) and 'other' (0)
    wake_word_index = list(feature_sets['target_words']).index(wake_word)
    y_train = np.equal(y_train, wake_word_index).astype('float64')
    y_val = np.equal(y_val, wake_word_index).astype('float64')
    y_test = np.equal(y_test, wake_word_index).astype('float64')

    # Peek at labels after conversion
    print(y_val)

    # What percentage of wake word appears in validation labels
    print("Appearance percentage:", 100 * sum(y_val) / len(y_val))

    return y_train, y_val, y_test


def save_performance_plots(history):
    """Save plot of performance."""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.close()


if __name__ == '__main__':
    """Train the model."""
    parser = argparse.ArgumentParser(description='Train tensorflow CNN model')
    parser.add_argument('-i', '--input', type=str, help='File of precomputed features')
    parser.add_argument('-o', '--output', type=str, help='Trained model file')
    parser.add_argument('-w', '--wake-word', type=str, help='Wake word to train to detect')
    parser.add_argument('--require-gpu', action='store_true', help='Error out if GPU unavailable')
    parser.add_argument('--save-plots', action='store_true', help='Whether to save performance plots')
    arguments = parser.parse_args()

    if arguments.require_gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Using {physical_devices[0]}")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            raise Exception("No GPU available")

    # Get inputs and outputs
    feature_sets = np.load(arguments.input)
    x_train, x_val, x_test = transform_inputs(feature_sets)
    y_train, y_val, y_test = transform_outputs(feature_sets, arguments.wake_word)

    # Create model
    sample_shape = x_train.shape[1:]
    model = create_model(sample_shape)

    # Train
    t0 = time.time()
    history = model.fit(x_train,
                        y_train,
                        epochs=30,
                        batch_size=100,
                        validation_data=(x_val, y_val))
    print(f"Total train time: {time.time() - t0}")

    # Plot results
    if arguments.save_plots:
        save_performance_plots(history)

    # Save the model as a file
    models.save_model(model, arguments.output)

    # Evaluate model
    loss, accuracy = model.evaluate(x=x_test, y=y_test)
    print(f"Evaluation loss = {loss}, accuracy = {accuracy}")
