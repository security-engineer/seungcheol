# POGER Dockerfile
# Based on: https://github.com/ICTMCG/POGER
# IJCAI 2024: Proxy-Guided Efficient Re-Sampling for Black-Box AI Text Detection
# Official requirements from requirements.txt

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Set python aliases
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 (from official requirements)
RUN python -m pip install torch==2.1.1+cu118 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install ALL dependencies from official requirements.txt
RUN python -m pip install \
    fschat==0.2.34 \
    httpx==0.27.0 \
    mosec==0.8.2 \
    msgpack==1.0.5 \
    nltk==3.8.1 \
    numpy==1.24.1 \
    openai==1.9.0 \
    scikit-learn==1.3.2 \
    tiktoken==0.5.2 \
    tqdm==4.66.1 \
    transformers==4.36.0 \
    sentencepiece \
    scipy \
    pandas

# Clone repository
RUN git clone https://github.com/ICTMCG/POGER.git /app/POGER

WORKDIR /app/POGER

# Create directories
RUN mkdir -p /app/models /app/data /app/POGER/get_feature

# ============================================================
# DATASETS: Download from Google Drive
# URL: https://drive.google.com/drive/folders/1xxdjZedn7le_P1HunCDF_WCuoFYI0-pz
# ============================================================

ENV PYTHONPATH=/app/POGER:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

CMD ["/bin/bash"]
