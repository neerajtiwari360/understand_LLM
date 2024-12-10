from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the model
llm = Llama(
    model_path="./Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=0,  # Set to 0 if running on CPU
)


class Item(BaseModel):
    prompt: str


@app.post("/predict")
async def predict(item: Item):
    prompt = item.prompt
    if not prompt:
        logging.info("No prompt provided")
        return {"error": "No prompt provided"}

    # Format the prompt according to the model schema
    formatted_prompt = f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>"

    logging.info(f"Received prompt: {formatted_prompt}")

    # Run the model
    output = llm(formatted_prompt, max_tokens=512, stop=["<|end|>"], echo=False)
    logging.info(f"Model output: {output}")

    # Extract usage and completion times
    usage = output.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # Add timings to the response
    response = {
        "response": output["choices"][0]["text"],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    return response
