# Start with a standard NVIDIA CUDA-enabled Python image.
# This ensures PyTorch can find the CUDA drivers on the host machine.
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set up the environment and install basic Python tools
ENV PYTHON_VERSION=3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.${PYTHON_VERSION} 1

# Install the correct PyTorch version for CUDA 11.8
# We install PyTorch first, which is critical for performance.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the remaining project files
COPY . /app

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]