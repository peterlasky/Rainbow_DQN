# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Set NVIDIA runtime requirements
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    swig \
    cmake \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy project files
COPY . .

# Set default command
CMD ["conda", "run", "-n", "myenv", "python", "main.py"]