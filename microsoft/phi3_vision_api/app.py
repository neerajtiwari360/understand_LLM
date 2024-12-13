import threading
import os
import glob
import time
from pathlib import Path
import json
from flask import Flask, request, jsonify
import onnxruntime_genai as og

app = Flask(__name__)

# Fixed values for model_path and provider
MODEL_PATH = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
PROVIDER = "cpu"

# Global result container to store results for requests
result_container = {}

def _find_dir_contains_sub_dir(current_dir: Path, target_dir_name):
    curr_path = Path(current_dir).absolute()
    target_dir = glob.glob(target_dir_name, root_dir=curr_path)
    if target_dir:
        return Path(curr_path / target_dir[0]).absolute()
    else:
        if curr_path.parent == curr_path:
            return None
        return _find_dir_contains_sub_dir(curr_path / '..', target_dir_name)


def _complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


def run(image_paths, prompt_text):
    print("Loading model...")
    if hasattr(og, 'Config'):
        config = og.Config(MODEL_PATH)
        config.clear_providers()
        if PROVIDER != "cpu":
            print(f"Setting model to {PROVIDER}...")
            config.append_provider(PROVIDER)
        model = og.Model(config)
    else:
        model = og.Model(MODEL_PATH)
    print("Model loaded")
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    images = None
    prompt = "<|user|>\n"

    if len(image_paths) == 0:
        print("No image provided")
    else:
        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            print(f"Using image: {image_path}")
            prompt += f"<|image_{i+1}|>\n"

        images = og.Images.open(*image_paths)

    prompt += f"{prompt_text}<|end|>\n<|assistant|>\n"
    print("Processing images and prompt...")
    inputs = processor(prompt, images=images)

    print("Generating response...")
    params = og.GeneratorParams(model)
    params.set_inputs(inputs)
    params.set_search_options(max_length=7680)

    generator = og.Generator(model, params)
    start_time = time.time()

    generated_text = ""
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        generated_text += tokenizer_stream.decode(new_token)

    total_run_time = time.time() - start_time
    print(f"Total Time : {total_run_time:.2f}")
    
    return generated_text


# A function to process the long-running task in a separate thread
def generate_in_thread(image_paths, prompt_text, result_container, task_id):
    response = run(image_paths, prompt_text)
    result_container[task_id] = response


@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()

        # Extract image_paths and prompt from the JSON body
        image_paths = data['image_paths']
        prompt_text = data['prompt']

        # Generate a unique task ID for the request
        task_id = str(time.time())  # Use a timestamp as a unique task ID

        # Create a new thread to handle the long-running task
        thread = threading.Thread(target=generate_in_thread, args=(image_paths, prompt_text, result_container, task_id))
        thread.start()

        # Return a quick acknowledgment with the task ID
        return jsonify({"message": "Processing started. The result will be available shortly.", "task_id": task_id}), 202

    except KeyError as e:
        return jsonify({'error': f'Missing parameter: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_result/<task_id>', methods=['GET'])
def get_result(task_id):
    # Retrieve the result using the task_id
    result = result_container.get(task_id)

    if result:
        return jsonify({'response': result}), 200
    else:
        return jsonify({'message': 'The processing is still ongoing.'}), 202


if __name__ == "__main__":
    # Run the Flask web server
    app.run(host='0.0.0.0', port=5000)
