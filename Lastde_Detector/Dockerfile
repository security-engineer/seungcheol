# Lastde_Detector Dockerfile
# Based on: https://github.com/TrustMedia-zju/Lastde_Detector
# ICLR 2025: Training-free LLM-generated Text Detection
# Official requirements from requirements.txt

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

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

# Install PyTorch with CUDA 11.8 (from official requirements)
RUN python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install ALL dependencies from official requirements.txt
RUN python -m pip install \
    accelerate==0.33.0 \
    datasets==2.21.0 \
    huggingface-hub==0.24.6 \
    matplotlib==3.7.5 \
    nltk==3.9.1 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    pyyaml==6.0.2 \
    safetensors==0.4.4 \
    scikit-learn==1.3.2 \
    scipy==1.10.1 \
    sentencepiece==0.2.0 \
    tokenizers==0.19.1 \
    tqdm==4.66.5 \
    transformers==4.44.1 \
    transformers-stream-generator==0.0.5

# Clone repository
RUN git clone https://github.com/TrustMedia-zju/Lastde_Detector.git /app/Lastde_Detector

WORKDIR /app/Lastde_Detector

# Create directories
RUN mkdir -p /app/models /app/pretrain_models

# ============================================================
# DATASETS: Included in datasets/ folder
# - human_original_data/, human_llm_data_for_experiment/
# - perturbation_data_detectgpt_npr/, regeneration_data_dnagpt/
# ============================================================

ENV PYTHONPATH=/app/Lastde_Detector:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

CMD ["/bin/bash"]
