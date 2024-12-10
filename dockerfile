# Use the official Python 3.10.12 slim image
FROM python:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first to take advantage of Docker's caching
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install llama-index

# Copy the application code
COPY app.py .

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
