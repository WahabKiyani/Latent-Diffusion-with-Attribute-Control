FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py .

# Create output directories
RUN mkdir -p outputs/results outputs/metrics

# Expose Gradio port
EXPOSE 7860

# Run application
CMD ["python3", "gradio_interface.py"]
