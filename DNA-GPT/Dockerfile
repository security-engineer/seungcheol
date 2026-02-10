# DNA-GPT Dockerfile
# Based on: https://github.com/Xianjun-Yang/DNA-GPT
# ICLR 2024: https://arxiv.org/abs/2305.17359
# Official requirements: rouge-score, transformers>=4.28.1, openai, sklearn, torch, nltk, spacy

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Set python aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ALL dependencies from official requirements.txt
RUN python -m pip install \
    numpy \
    scipy \
    pandas \
    rouge-score \
    transformers>=4.28.1 \
    openai \
    scikit-learn \
    nltk \
    spacy \
    gradio \
    tqdm \
    sentencepiece

# Clone repository
RUN git clone https://github.com/Xianjun-Yang/DNA-GPT.git /app/DNA-GPT

WORKDIR /app/DNA-GPT

# Create directories
RUN mkdir -p /app/models /app/data

# ============================================================
# DATASETS: Use get_data/ scripts to generate datasets
# ============================================================

ENV PYTHONPATH=/app/DNA-GPT:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 7860
CMD ["python", "DNA-GPT-dist.py"]
