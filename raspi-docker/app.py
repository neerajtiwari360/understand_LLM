from llama_cpp import Llama
import sys
import os
import readline

# ANSI color codes for colored terminal output
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Suppress verbose output from llama_cpp
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Initialize the Llama model
llm = Llama(
  model_path="/app1/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
  n_ctx=2048,
  n_threads=8,
  n_gpu_layers=0,
  verbose=False  # Disable verbose logging
)

# Restore stdout and stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Define your system message
system_message = "You are a helpful assistant. Keep your responses short and concise"

# Loop for continuous interaction
while True:
    # Get user input with yellow color
    prompt = input(f"{YELLOW}You: {RESET}")

    # Exit the loop if the user types "exit"
    if prompt.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    # Construct the prompt for the model
    full_prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"

    # Suppress verbose output during inference
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # Generate a streaming response
    stream = llm(
        full_prompt,  # Prompt
        max_tokens=512,  # Generate up to 512 tokens
        stop=["</s>"],   # Stop token
        echo=False,      # Do not echo the prompt
        stream=True      # Enable streaming
    )

    # Restore stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    # Print Assistant response inline
    print(f"{BLUE}Assistant:{RESET} ", end="", flush=True)
    # Stream and print each token as it's generated
    for output in stream:
        token = output["choices"][0]["text"].replace("\n", " ")
        print(token, end="", flush=True)

    print("\n")  # Print newline after response completion
