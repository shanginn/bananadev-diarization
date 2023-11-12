# Base image for Banana model builds
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Accept HF_TOKEN as a build-time argument
ARG HF_TOKEN

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the HF_TOKEN environment variable
ENV HF_TOKEN=${HF_TOKEN}

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip typing-extensions -r requirements.txt

# Add and download your model weight files
COPY download.py .
RUN python3 download.py

# Add the rest of your code
COPY . .

EXPOSE 8000

# Start the app in the container
CMD ["python3", "-u", "app.py"]
