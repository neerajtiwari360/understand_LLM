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
#RUN huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir . --local-dir-use-symlinks False
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

# Copy the requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

#RUN curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3v.py -o phi3v.py
RUN pip install pyreadline3
RUN pip install onnxruntime-genai
RUN pip install flask

# Copy the downloaded model from the builder stage
COPY --from=builder /app .
COPY Phi-3.png /app/Phi-3.png

# Copy the rest of the application
COPY . /app

# Set environment variables
ENV NAME World

# Expose the port the app runs on
EXPOSE 5000

CMD ["python", "app.py"]

# # Define the command to run the app
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]