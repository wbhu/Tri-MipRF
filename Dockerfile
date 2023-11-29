FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Update and install some essential packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3 (Ubuntu 22.04 comes with Python 3.10)
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory inside the container
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN TCNN_CUDA_ARCHITECTURES=89 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install nvdiffrast: https://nvlabs.github.io/nvdiffrast/#linux

RUN pip3 install --no-cache-dir -r requirements.txt