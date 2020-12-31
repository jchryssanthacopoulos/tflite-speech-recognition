# TensorFlow Lite Speech Recognition

## Installation

### Training

Training usually involves a GPU desktop computer.

In a virtual environment, run

```bash
pip install -r requirements_train.txt
```

The installation was *only* tested with:

1. Ubuntu 18.04.5 LTS (Bionic Beaver)
2. Python 3.8.5 (see `.python-version`)

I had to run the following command to get the dependencies to install:

```bash
sudo apt-get install pkg-config libcairo2-dev gcc python3-dev libgirepository1.0-dev
```

### Deploy

Deployment usually involves a single-board computer like the Raspberry Pi.

In a virtual environment, run

```bash
pip install -r requirements_deploy.txt
```

The installation was *only* tested with:

1. Raspberry Pi 3B+ running Bionic Beaver
2. Python 3.8.5

I had to run the following command to get the dependencies to install:

```bash
sudo apt-get install libblas-dev gfortran libopenblas-base libatlas-base-dev
```
