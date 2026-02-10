# DetectGPT Dockerfile
# Based on: https://github.com/eric-mitchell/detect-gpt
# ICML 2023: Zero-Shot Machine-Generated Text Detection using Probability Curvature
#
# ============================================================
# OFFICIAL REQUIREMENTS (from README.md):
# - pip install -r requirements.txt
# - data/writingPrompts/ 폴더에 데이터 필요 (Kaggle 다운로드)
# ============================================================

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
    transformers \
    datasets \
    matplotlib \
    tqdm \
    scikit-learn \
    openai \
    nltk \
    sentencepiece

# Clone repository
RUN git clone https://github.com/eric-mitchell/detect-gpt.git /app/detect-gpt

WORKDIR /app/detect-gpt

# Create directories
RUN mkdir -p /app/models /app/data /app/data/writingPrompts /app/results /app/tmp_results

# ============================================================
# DATASETS: Download WritingPrompts from Kaggle
# URL: https://www.kaggle.com/datasets/ratthachat/writing-prompts
# Save to: /app/data/writingPrompts/
# ============================================================

# HuggingFace cache settings to avoid symlink issues in Docker
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_CACHE=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV XDG_CACHE_HOME=/tmp

# Fix run.py to expand ~ to absolute path (prevents symlink issues)
RUN sed -i 's|cache_dir = args.cache_dir|cache_dir = os.path.expanduser(args.cache_dir)|g' /app/detect-gpt/run.py

# Create cache directory with proper permissions
RUN mkdir -p /tmp/hf_cache && chmod 777 /tmp/hf_cache

ENV PYTHONPATH=/app/detect-gpt:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

CMD ["/bin/bash"]
