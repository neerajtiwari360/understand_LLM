from flask import Flask, request, jsonify
import onnxruntime_genai as og

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model = og.Model('cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')  # Replace with your model path
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set search options
search_options = {}
search_options['max_length'] = 2048

# Initialize the message history with system message
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"}
]

# Define the chat template
chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

# Function to handle the user input and generate a response
def chatbot_response(text):
    if not text:
        return "Error, input cannot be empty"
    
    # Append the user's message to the message history
    messages.append({"role": "user", "content": text})
    
    # Format the conversation for the prompt
    prompt = f'{chat_template.format(input=text)}'
    input_tokens = tokenizer.encode(prompt)
    
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    response = ""
    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            response += tokenizer_stream.decode(new_token)
    except KeyboardInterrupt:
        response += "  --control+c pressed, aborting generation--"
    
    # Add the bot's response to the message history
    messages.append({"role": "assistant", "content": response.strip()})
    
    return response.strip()

# Define Flask route for the chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    # Call the chatbot_response function to generate the response
    response = chatbot_response(user_input)
    return jsonify({"response": response})

# Run the Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
