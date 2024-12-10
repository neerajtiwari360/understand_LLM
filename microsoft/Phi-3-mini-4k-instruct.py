from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use float32 for CPU
    device_map="cpu",  # Explicitly set to CPU
)

# Initialize conversation history
messages = []

@app.route("/chat", methods=["POST"])
def chat():
    global messages

    # Get user input from the request
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Append the user's message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Tokenize input
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"  # Keep tensors on CPU
    )

    # Define terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate outputs
    outputs = model.generate(
        input_ids,
        max_new_tokens=32,  # Adjust for performance
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Decode the response
    response = outputs[0][input_ids.shape[-1]:]
    bot_response = tokenizer.decode(response, skip_special_tokens=True)

    # Add the bot's response to the conversation history
    messages.append({"role": "assistant", "content": bot_response.strip()})

    return jsonify({"response": bot_response.strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
