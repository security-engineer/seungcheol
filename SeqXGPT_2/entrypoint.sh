#!/bin/bash
set -e

# If SeqXGPT source code is missing (volume mount overwrites build-time files),
# clone it into the mounted host directory so it persists
if [ ! -d "/app/SeqXGPT/.git" ]; then
    echo "==> SeqXGPT source not found. Cloning into /app/SeqXGPT ..."
    git clone https://github.com/Jihuai-wpy/SeqXGPT.git /app/SeqXGPT
    echo "==> Clone complete."
else
    echo "==> SeqXGPT source already present."
fi

# Create necessary directories
mkdir -p /app/models /app/data /app/datasets

exec "$@"
