# Phi 3 Dockerized Deployment

This repository provides a streamlined approach to deploy the quantized [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) model using Docker, enabling efficient backend setup for text generation tasks.

## Project Highlights

This project sets up a lightweight FastAPI server that provides an endpoint to generate text using the Phi-3-mini-4k-instruct model. The setup is minimalist yet extensible, allowing you to tailor the API in `app.py` for custom use cases.

---

## Quick Start

### Run the Prebuilt Docker Image

For immediate setup and usage, simply pull and run the Docker container:
```bash
docker run -p 4000:5000 chatbot-api
```

### Send a Prompt Request
Use `curl` or similar tools to send a request to the running server:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt":"How to explain the Internet to a medieval knight?"}' http://localhost:4000/predict
```

---

## Prerequisites

1. **Docker**: Ensure Docker is installed on your system. Instructions can be found [here](https://docs.docker.com/get-docker/).
2. **Hugging Face Token**: To build the Docker image with the model, you’ll need an authentication token from [Hugging Face](https://huggingface.co).

---

## Setup

### Build the Image Locally
This option is ideal if you want to customize the image or the model parameters.

#### 1. Clone the Repository
```bash
git clone https://github.com/neerajtiwari360/understand_LLM.git
cd understanding_LLM/raspi-docker
```

#### 2. Configure Model Parameters
Adjust the parameters in `app.py` to suit your system:
```python
llm = Llama(
    model_path="./Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,  # Max context length. Shorten for lower memory usage.
    n_threads=8,  # Number of CPU threads.
    n_gpu_layers=0  # Set to 0 for CPU-only inference, or -1 for full GPU inference.
)
```

#### 3. Build the Docker Image
Pass your Hugging Face token as a build argument:
```bash
docker build --build-arg HF_AUTH_TOKEN=your_hugging_face_token -t chatbot-api .
```

#### 3.3 Download the tiny llama model from `https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
- and keep it into `/home/neeraj/model`

#### 4. Run the Container
```bash
docker run -it --rm -p 4000:5000 -v /home/neeraj/model:/app1 chatbot-api /bin/bash
$ python app.py
```

---