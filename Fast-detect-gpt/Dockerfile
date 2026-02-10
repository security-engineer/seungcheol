# Fast-DetectGPT Dockerfile
# Based on: https://github.com/baoguangsheng/fast-detect-gpt
# ICLR 2024: Efficient Zero-Shot Detection via Conditional Probability Curvature
#
# ============================================================
# OFFICIAL REQUIREMENTS (from README.md):
# - Python3.8
# - PyTorch1.10.0
# ============================================================

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Set python aliases
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 1.10.0 with CUDA 11.3 (OFFICIAL REQUIREMENT)
RUN python -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install ALL dependencies from official requirements.txt + sklearn for metrics.py
RUN python -m pip install \
    numpy \
    scipy \
    scikit-learn \
    transformers==4.28.1 \
    datasets==2.12.0 \
    matplotlib \
    tqdm \
    openai \
    nltk \
    sentencepiece

# Clone repository
RUN git clone https://github.com/baoguangsheng/fast-detect-gpt.git /app/fast-detect-gpt

WORKDIR /app/fast-detect-gpt

# Create directories
RUN mkdir -p /app/models /app/data

# ============================================================
# DATASETS: exp_gpt3to4/data/ contains GPT-3/3.5/4 generated text
# ============================================================

ENV PYTHONPATH=/app/fast-detect-gpt:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 7860
WORKDIR /workspace
CMD ["/bin/bash"]
