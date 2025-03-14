from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(
  model_path="/app1/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
  n_ctx=2048,
  n_threads=8,
  n_gpu_layers=0
)

# Define your system message and user prompt
system_message = "You are a helpful assistant."
prompt = "Tell me a joke."

# Simple inference example
output = llm(
  f"<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>",  # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  stop=["</s>"],   # Stop token
  echo=True        # Whether to echo the prompt
)

# Extract and print the generated text
generated_text = output["choices"][0]["text"]
print(generated_text)