# Stage 1: Build stage with secret to download the model
FROM ubuntu:22.04 AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir huggingface-hub>=0.17.1

# Set the environment variable for the huggingface-cli
ENV HUGGINGFACE_HUB_ADD_TOKEN_AS_GIT_CREDENTIAL=false

# Use ARG to pass the secret token for use only during the build
ARG HF_AUTH_TOKEN

# Log in to Hugging Face and download the model
RUN huggingface-cli login --token $HF_AUTH_TOKEN
RUN huggingface-cli download microsoft/Phi-3.5-vision-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .

# Stage 2: Final image without secrets
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Install system dependencies for Python and pip
RUN apt-get update && apt-get install -y bash \
    python3 \
    python3-pip \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for the application
RUN pip3 install --no-cache-dir pyreadline3 onnxruntime-genai

# Copy the downloaded model from the builder stage
COPY --from=builder /app .

# Copy the rest of the application files
COPY . /app

# Copy the test image
COPY australia.jpg /app/australia.jpg

# Set environment variables
ENV NAME World

# Expose the port the app runs on
EXPOSE 5000

# Default command to run the application
CMD ["python3", "app.py", "-m", "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4", "-p", "cpu", "--image_paths", "australia.jpg", "--non-interactive", "--prompt", "What is shown in this image?"]
