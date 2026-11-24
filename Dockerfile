# SoundSense — Environmental Sound Classification Studio
# Optimized for Hugging Face Spaces (Docker SDK), Railway, and Cloud Run

FROM python:3.11-slim

# Install system audio dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    RELOAD=false

WORKDIR $HOME/app

# Install Python dependencies with CPU-optimized PyTorch
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . $HOME/app

# Ensure uploads directory is writeable
RUN mkdir -p $HOME/app/uploads

# Pre-fetch AST weights so the container is 100% self-contained and boots instantaneously
RUN python -m eval.fetch_ast || true

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Start FastAPI server
CMD ["python", "main.py"]
