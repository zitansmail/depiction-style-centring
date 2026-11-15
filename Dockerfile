# GPU image (NVIDIA driver + nvidia-container-toolkit on the host); also runs on CPU, more slowly.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git curl unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/app/cache/huggingface
# default: run the CPU tests; override with e.g. `python scripts/reproduce.py all`
CMD ["python", "-m", "pytest", "-q", "tests"]
