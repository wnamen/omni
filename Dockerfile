# Python backend for OmniParser service
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt ./

# Install dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Clone OmniParser repository
RUN git clone https://github.com/microsoft/OmniParser.git

# Copy server application
COPY server/ ./server/

# Copy .env file if it exists (optional)
COPY .env* ./

# Install OmniParser requirements
RUN pip install -r OmniParser/requirements.txt

# Install transformers
RUN pip install transformers==4.49.0

# Create weights directory
RUN mkdir -p weights

# Add OmniParser to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/OmniParser"

# Set Python optimization flags
ENV PYTHONOPTIMIZE=1
ENV PYTHONDONTWRITEBYTECODE=1

# Download model weights (pre-download during build)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/OmniParser-v2.0', local_dir='weights')"

# Pre-download Florence-2-base model
RUN python -c "from transformers import AutoProcessor, AutoModelForCausalLM; processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', torch_dtype='auto', trust_remote_code=True)"

# Confirm all models are downloaded and accessible
RUN python -c "import os; print('OmniParser-v2.0 model weights exist:', os.path.exists('weights/icon_detect/model.pt')); print('Florence-2-base model downloaded successfully')"

# Remove __pycache__ directories to prevent memory issues
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Expose the port the app runs on
EXPOSE 2171

# Command to run the application with auto-reload, excluding problematic directories
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "2171", "--reload", "--reload-exclude", "*/__pycache__/*", "--reload-exclude", "*/.*"] 