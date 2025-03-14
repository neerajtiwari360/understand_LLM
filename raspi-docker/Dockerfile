# Stage 1: Build stage with secret to download the model
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install huggingface-hub>=0.17.1

# Set the environment variable for the huggingface-cli
ENV HUGGINGFACE_HUB_ADD_TOKEN_AS_GIT_CREDENTIAL=false

# Use ARG to pass the secret token for use only during the build
ARG HF_AUTH_TOKEN

# Log in to Hugging Face and download the model
RUN apt-get update && apt-get install -y git
RUN huggingface-cli login --token $HF_AUTH_TOKEN
#RUN huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir . --local-dir-use-symlinks False

# Stage 2: Final image without secrets
FROM python:3.11-slim

WORKDIR /app

# ✅ Install required system dependencies for Raspberry Pi (ARM)
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip and setuptools to avoid dependency issues
RUN pip install --upgrade pip setuptools wheel

# ✅ Force installation of precompiled NumPy wheels (ARM-compatible)
RUN pip install --prefer-binary --no-cache-dir "numpy<2.0"

# ✅ Try to install `llama-cpp-python` using prebuilt wheels
RUN pip install --no-cache-dir --prefer-binary "llama-cpp-python"

# ✅ If that fails, manually compile `llama-cpp-python` for ARM
RUN CMAKE_ARGS="-DLLAMA_NATIVE=OFF -DLLAMA_ARCH=ARM64" pip install --no-cache-dir "llama-cpp-python"

# ✅ Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the downloaded model from the builder stage
COPY --from=builder /app .

# Copy the rest of the application
COPY . /app

# Set environment variables
ENV NAME World

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "5000"]
