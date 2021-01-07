# TensorFlow Lite Speech Recognition

## Installation

For both training and deployment, Python 3.7.9 is required. If you're using pyenv, the Python version will be set automatically by `.python-version`.

### Training

Training usually involves a GPU desktop computer.

In a virtual environment, run

```bash
pip install -r requirements_train.txt
```

The installation was *only* tested with Ubuntu 18.04.5 LTS (Bionic Beaver).

I had to run the following command to get the dependencies to install:

```bash
sudo apt-get install pkg-config libcairo2-dev gcc python3-dev libgirepository1.0-dev
```

### Using Docker

1. Install Docker, etc.
2. Pull latest tensorflow GPU image: `docker pull tensorflow/tensorflow:latest-gpu-jupyter`
3. Build image on top of base: `docker build -t tflite-speech-recognition .`
4. Run: `./train_model_gpu.sh` to train the model within a GPU-enabled container and return the model file back to host.

### Deploy

Deployment usually involves a single-board computer like the Raspberry Pi.

In a virtual environment, run

```bash
pip install -r requirements_deploy.txt
```

The installation was *only* tested with Raspberry Pi 3B+, again running Bionic Beaver.

I had to run the following command to get the dependencies to install:

```bash
sudo apt-get install libblas-dev gfortran libopenblas-base libatlas-base-dev libgfortran5
```
