# ==============================================================================
# Dockerfile for Latent Diffusion with Attribute Control
# Supports GPU acceleration with CUDA 12.6
# ==============================================================================

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set environment variables
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

# Set working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.6 support
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Copy requirements file
COPY requirements.txt .

# Install remaining dependencies (PyTorch already installed above)
RUN pip3 install --no-cache-dir \
    xformers \
    diffusers>=0.35.2 \
    transformers>=4.57.0 \
    accelerate>=1.12.0 \
    opencv-python \
    mediapipe \
    controlnet-aux \
    gradio>=6.0.0 \
    omegaconf \
    peft \
    safetensors \
    torchmetrics \
    torch-fidelity \
    scikit-image \
    scipy \
    numpy \
    matplotlib \
    Pillow \
    huggingface-hub

# Copy project files
COPY Proj_enhanced.py .
COPY pretrained/ ./pretrained/

# Create output directories
RUN mkdir -p outputs/results outputs/metrics outputs/visualizations

# Expose Gradio port
EXPOSE 7860

# Set Python path
ENV PYTHONPATH=/app

# Run the application
CMD ["python3", "Proj_enhanced.py"]
