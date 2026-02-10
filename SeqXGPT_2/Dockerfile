# SeqXGPT Dockerfile
# Based on: https://github.com/Jihuai-wpy/SeqXGPT
# ArXiv 2023: Sentence-Level AI-Generated Text Detection

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip python3.8-distutils \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.3
RUN python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install core dependencies (from analyzing imports in the project)
RUN python -m pip install \
    numpy==1.21.6 \
    scipy==1.7.3 \
    pandas==1.3.5 \
    scikit-learn==1.0.2 \
    matplotlib==3.5.3 \
    tqdm==4.64.1 \
    pyyaml==6.0 \
    requests==2.28.1

# Install NLP dependencies
RUN python -m pip install \
    nltk==3.6.2 \
    spacy==3.4.4 \
    regex==2022.10.31 \
    ftfy==6.1.1

# Install Hugging Face ecosystem (Llama 3.1 compatible)
RUN python -m pip install \
    transformers==4.43.4 \
    tokenizers==0.19.1 \
    datasets==2.10.0 \
    huggingface-hub==0.24.6 \
    accelerate==0.28.0 \
    sentencepiece==0.1.95

# Install deep learning utilities
RUN python -m pip install \
    einops==0.6.0 \
    deepspeed==0.8.1

# Install API server dependencies
RUN python -m pip install \
    flask==2.2.2 \
    werkzeug==2.2.2 \
    mosec==0.6.0 \
    msgpack==1.0.5

# Install OpenAI dependencies
RUN python -m pip install \
    openai==0.27.2 \
    httpx==0.23.3 \
    tiktoken

# Install logging tools
RUN python -m pip install \
    wandb==0.13.5 \
    tensorboard==2.8.0 \
    tensorboardX==2.5.1 \
    seqeval==1.2.2 \
    fire==0.5.0 \
    jsonlines \
    filelock==3.8.0

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app

ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 6006
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
