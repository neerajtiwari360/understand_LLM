# Stage 2: Final image without secrets
FROM python:3.11-slim

WORKDIR /app

# Install necessary tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir torch transformers huggingface-hub>=0.17.1

RUN pip install accelerate>=0.26.0

# Set the environment variable for the huggingface-cli
ENV HUGGINGFACE_HUB_ADD_TOKEN_AS_GIT_CREDENTIAL=false

# Use ARG to pass the secret token for use only during the build
ARG HF_AUTH_TOKEN

# Log in to Hugging Face and download the model
RUN huggingface-cli login --token $HF_AUTH_TOKEN

# Copy the model from the builder stage
#COPY --from=builder /app/model /app/model

# Copy the Python script to the container
COPY app.py /app/script.py

# Expose the port (optional, in case you expose an API)
# EXPOSE 5000

# Define the command to run the script
CMD ["python", "/app/script.py"]
