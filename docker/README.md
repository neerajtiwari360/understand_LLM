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
cd docker
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

#### 4. Run the Container
```bash
docker run -p 4000:5000 chatbot-api
```

---

## Using the API

The API exposes a single endpoint to interact with the model:

### Endpoint Details

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Request Format
The request body should be a JSON object with a `prompt` field:
```json
{
  "prompt": "How to explain the Internet to a medieval knight?"
}
```

### Response Format
The response will include the generated text and token usage:
```json
{
  "response": "The Internet is like a network of magical scrolls carrying messages between kingdoms, powered by invisible forces.",
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 32,
    "total_tokens": 48
  }
}
```

---

## Example Request Using `curl`
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"prompt":"What is the purpose of life?"}' \
     http://localhost:4000/predict
```

---

## Acknowledgments

- **Microsoft Research**: For developing the Phi-3 model ([more info](https://www.microsoft.com/en-us/research/publication/phi-3-technical-report-a-highly-capable-language-model-locally-on-your-phone/)).
- **Hugging Face**: For hosting and providing the model ([Phi-3 on Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)).
- **FastAPI**: For the lightweight web framework.
- **Llama Library**: Used for model integration ([repo link](https://github.com/yourusername/llama-cpp)).

---

## Additional Resources

To better understand the deployment process and how to maximize Phi-3’s capabilities, check out the following resources:

- **Blog**: Learn the complete setup and the benefits of small LLMs in this detailed [Medium blog post](https://medium.com/@neeraztiwari/phi-3-deploying-compact-ai-models-for-real-world-applications-8e03cdcac5cd).
- **YouTube Video**: Watch the step-by-step guide on [YouTube](https://youtu.be/jeAOcyp8yhg?si=CkdXS83Gddn8uoPV).

---

## Responsible AI Practices

Phi-3 is a highly capable model, but like all large language models, it has inherent limitations and risks:

### Key Considerations
1. **Bias and Representation**: Models may perpetuate societal biases reflected in the training data.
2. **Language Performance**: Non-English and non-standard dialects may see reduced accuracy.
3. **Harmful Content**: Despite mitigation efforts, the model may produce inappropriate or offensive responses.
4. **Information Reliability**: Outputs may include inaccurate or fabricated information.

### Recommendations
- Clearly inform users they are interacting with AI-generated content.
- Implement feedback mechanisms for corrections and improvements.
- Avoid deploying the model in high-risk scenarios without additional safeguards.

For more details, visit the [Responsible AI Considerations](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#responsible-ai-considerations) section on Hugging Face.
