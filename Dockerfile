# Use a PyTorch base image with CUDA 12.1 support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

LABEL maintainer="tasal9" \
      description="Environment for fine-tuning and inference with ZamAI Phi-3 Pashto model."

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train_lora.py", "--help"]
