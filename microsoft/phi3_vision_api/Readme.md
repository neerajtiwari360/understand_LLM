# Phi3 ONNX API

This Flask-based API processes images and generates text responses based on image content using an ONNX model.

## Prerequisites

- Python 3.12+
- `pip` for package management
- Docker (optional)

## Installation

1. **Build the Docker image**:

   First, build the Docker image with the following command:

   ```bash
   sudo docker build --build-arg HF_AUTH_TOKEN=<HF TOKEN> -t phi_vision .
   ```

2. **Run the Docker container**:

   Start the container with the following command:

   ```bash
   sudo docker run -it --rm -p 4000:5000 phi_vision
   ```

3. **Ensure the ONNX model and images are available**:

   Make sure that the ONNX model (`cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4`) and any required images are available in the correct directory.

## Running the API

1. **Start the Flask server**:

   If you're running without Docker, start the Flask server with the following command:

   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:4000`.

## API Endpoints

### `/generate` (POST)
This endpoint starts processing an image and prompt.

#### Request Body:

```json
{
  "image_paths": ["Phi-3.png"],
  "prompt": "What is shown in this image?"
}
```

#### Response:

```json
{
  "message": "Processing started.",
  "task_id": "unique_task_id"
}
```

### `/get_result/<task_id>` (GET)
This endpoint retrieves the result of a task using the provided `task_id`.

#### Response:

- **Processing ongoing**:

```json
{
  "message": "The processing is still ongoing."
}
```

- **Task complete**:

```json
{
  "response": "Generated response text"
}
```

## Running with Docker (Optional)

If you prefer to run the application using Docker:

1. **Build the Docker image**:

   ```bash
   docker build -t onnx-image-api .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 5000:5000 onnx-image-api
   ```

   The Flask server will be available at `http://localhost:4000`.
