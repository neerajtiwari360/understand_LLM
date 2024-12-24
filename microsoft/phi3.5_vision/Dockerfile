# Stage 1: Build stage with secret to download the model
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install huggingface-hub>=0.17.1

# Set the environment variable for the huggingface-cli
ENV HUGGINGFACE_HUB_ADD_TOKEN_AS_GIT_CREDENTIAL=false

# Use ARG to pass the secret token for use only during the build
ARG HF_AUTH_TOKEN

# Log in to Hugging Face and download the model
RUN huggingface-cli login --token $HF_AUTH_TOKEN
RUN huggingface-cli download microsoft/Phi-3.5-vision-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .

# Stage 2: Final image without secrets
FROM python:3.12-slim

WORKDIR /app

# Install dependencies required for the build
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install onnxruntime-genai
RUN pip install gradio

# Copy the downloaded model from the builder stage
COPY --from=builder /app .
COPY app.py /app/app.py

# Set environment variables
ENV NAME World

# Expose the port the app runs on
EXPOSE 4050

CMD ["python", "app.py"]