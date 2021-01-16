# TensorFlow Lite Speech Recognition

## Installation

For both training and deployment, Python 3.7.9 is required. If you're using pyenv, the Python version will be set automatically by `.python-version`.

### Training

Training usually involves a GPU desktop computer using Docker.

1. In a virtual environment, run

```bash
pip install -r requirements_train.txt
```

2. Copy `.env.sample` to `.env`, then fill out.

3. Pull the latest Tensorflow GPU image:

```bash
: `docker pull tensorflow/tensorflow:latest-gpu-jupyter`
```

4. Run the script to extract audio features, and train and save the Tensorflow Lite model:

```bash
chmod o+x train.sh
./train.sh
```

This program was *only* tested with Ubuntu 18.04.5 LTS (Bionic Beaver).

I had to run the following command to get the dependencies to install:

```bash
sudo apt-get install pkg-config libcairo2-dev gcc python3-dev libgirepository1.0-dev
```

### Deploy

Deployment usually involves a single-board computer like the Raspberry Pi.

1. In a virtual environment, run

```bash
pip install -r requirements_deploy.txt
```

2. Run the script to detect words:

```bash
python detect_multi_word.py -i <model_lite_file>
```

where `model_lite_file` is the trained model.

This program was *only* tested with Raspberry Pi 3B+, again running Bionic Beaver.

I had to run the following command to get the dependencies to install:

```bash
sudo apt-get install libblas-dev gfortran libopenblas-base libatlas-base-dev libgfortran5
```
