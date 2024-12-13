import gradio as gr
import os
import time
import onnxruntime_genai as og

# Global variable to hold the model
model = None

def generate_response(image, prompt_text, provider):
    # Ensure the model is loaded once
    global model

    # If the image is provided, process it
    prompt = "<|user|>\n"
    if image:
        try:
            image_path = image  # Single image path, so no need for a loop
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            print(f"Using image: {image_path}")
            prompt += f"<|image_1|>\n"

            images = og.Images.open(image_path)
        except Exception as e:
            return f"Error loading image: {e}"
    else:
        images = None

    prompt += f"{prompt_text}<|end|>\n<|assistant|>\n"
    print("Processing images and prompt...")
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()
    inputs = processor(prompt, images=images)

    # Generate response
    print("Generating response...")
    params = og.GeneratorParams(model)
    params.set_inputs(inputs)
    params.set_search_options(max_length=7680)

    generator = og.Generator(model, params)
    start_time = time.time()

    response = ""
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()

        new_token = generator.get_next_tokens()[0]
        response += tokenizer_stream.decode(new_token)

    total_run_time = time.time() - start_time
    print(f"Total Time: {total_run_time:.2f}")
    return response


# Gradio interface
def gradio_interface(model_path, provider):
    global model

    # Load the model once, outside the Gradio interface
    if model is None:
        print("Loading model...")
        if hasattr(og, 'Config'):
            config = og.Config(model_path)
            config.clear_providers()
            if provider != "cpu":
                print(f"Setting model to {provider}...")
                config.append_provider(provider)
            model = og.Model(config)
        else:
            model = og.Model(model_path)
        print("Model loaded.")

    # Gradio inputs for image and prompt
    image_input = gr.Image(type="filepath", label="Upload Image")
    prompt_input = gr.Textbox(label="Enter your prompt", lines=2)

    # Function for processing the inputs and generating the response
    def process_image_and_prompt(image, prompt_text):
        response = generate_response(image, prompt_text, provider)
        return response

    # Create and launch the Gradio interface
    interface = gr.Interface(
        fn=process_image_and_prompt,
        inputs=[image_input, prompt_input],
        outputs="text",
        title="Phi3 Vision CPU",
        description="Upload an image, enter a prompt, and get a response generated from the model."
    )
    
    interface.launch(server_name="0.0.0.0", server_port=4050)


# Example: Initialize Gradio interface with your model path and provider
if __name__ == "__main__":
    model_path = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"  # Replace with actual model path
    provider = "cpu"  # Replace with the desired provider ("cpu", "cuda", etc.)
    gradio_interface(model_path, provider)





# import gradio as gr
# import os
# import glob
# import time
# from pathlib import Path
# import onnxruntime_genai as og

# def _find_dir_contains_sub_dir(current_dir: Path, target_dir_name):
#     curr_path = Path(current_dir).absolute()
#     target_dir = glob.glob(target_dir_name, root_dir=curr_path)
#     if target_dir:
#         return Path(curr_path / target_dir[0]).absolute()
#     else:
#         if curr_path.parent == curr_path:
#             # Root dir
#             return None
#         return _find_dir_contains_sub_dir(curr_path / '..', target_dir_name)

# def generate_response(image, prompt_text, model_path, provider):
#     # Initialize the model
#     print("Loading model...")
#     if hasattr(og, 'Config'):
#         config = og.Config(model_path)
#         config.clear_providers()
#         if provider != "cpu":
#             print(f"Setting model to {provider}...")
#             config.append_provider(provider)
#         model = og.Model(config)
#     else:
#         model = og.Model(model_path)
#     print("Model loaded")
    
#     # Create processor and tokenizer stream
#     processor = model.create_multimodal_processor()
#     tokenizer_stream = processor.create_stream()

#     # Prepare the prompt and image for the model
#     prompt = "<|user|>\n"
#     if image is not None:
#         image_paths = [image]
#         for i, image_path in enumerate(image_paths):
#             if not os.path.exists(image_path):
#                 raise FileNotFoundError(f"Image file not found: {image_path}")
#             print(f"Using image: {image_path}")
#             prompt += f"<|image_{i+1}|>\n"

#         images = og.Images.open(*image_paths)
#     else:
#         images = None

#     prompt += f"{prompt_text}<|end|>\n<|assistant|>\n"
#     print("Processing images and prompt...")
#     inputs = processor(prompt, images=images)

#     # Generate response
#     print("Generating response...")
#     params = og.GeneratorParams(model)
#     params.set_inputs(inputs)
#     params.set_search_options(max_length=7680)

#     generator = og.Generator(model, params)
#     start_time = time.time()

#     response = ""
#     while not generator.is_done():
#         generator.compute_logits()
#         generator.generate_next_token()

#         new_token = generator.get_next_tokens()[0]
#         response += tokenizer_stream.decode(new_token)

#     total_run_time = time.time() - start_time
#     print(f"Total Time : {total_run_time:.2f}")

#     return response


# # Gradio interface
# def gradio_interface(model_path, provider):
#     # Gradio inputs for image and prompt
#     image_input = gr.Image(type="filepath", label="Upload Image")  # Fixed 'type' to 'filepath'
#     prompt_input = gr.Textbox(label="Enter your prompt", lines=2)

#     # Function for processing the inputs and generating the response
#     def process_image_and_prompt(image, prompt_text):
#         response = generate_response(image, prompt_text, model_path, provider)
#         return response

#     # Create and launch the Gradio interface
#     interface = gr.Interface(
#         fn=process_image_and_prompt, 
#         inputs=[image_input, prompt_input],
#         outputs="text",
#         title="Image and Prompt Processor",
#         description="Upload an image, enter a prompt, and get a response generated from the model."
#     )
    
#     interface.launch(server_name="0.0.0.0", server_port=4050, share=True)

# # Example: Initialize Gradio interface with your model path and provider
# if __name__ == "__main__":
#     model_path = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"  # Replace with actual model path
#     provider = "cpu"  # Replace with the desired provider ("cpu", "cuda", etc.)
#     gradio_interface(model_path, provider)
