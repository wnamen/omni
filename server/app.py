import logging
import sys
import time
import os
import gc
import base64
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import torch
from PIL import Image
import io


from server.util.omniparser import Omniparser

# Configure logging with timestamp format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()
omniparser = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def cleanup_memory():
    """Explicitly clean up memory and force garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global omniparser

    try:
        logger.info(f"Initializing models on device: {device}")
        
        config = {
            'som_model_path': os.path.join(os.getcwd(), 'weights/icon_detect/model.pt'),
            'caption_model_name': 'microsoft/Florence-2-base',
            'caption_model_path': os.path.join(os.getcwd(), 'weights/icon_caption_florence'),
            'BOX_TRESHOLD': 0.05,
            'device': device
        }

        omniparser = Omniparser(config)
        
        # Clean up after model loading
        cleanup_memory()
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        # Don't raise the exception to allow the server to start anyway
        # but log the error for debugging
        logger.exception("Model initialization failed")


@app.get("/api/health")
async def health_check():
    global omniparser

    models_loaded = omniparser is not None
    cuda_ok = True
    model_ok = models_loaded
    cuda_memory_allocated = 0
    cuda_memory_reserved = 0

    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            cuda_memory_allocated = torch.cuda.memory_allocated()
            cuda_memory_reserved = torch.cuda.memory_reserved()
        except Exception:
            cuda_ok = False

    # Optionally, do a dummy check to ensure models are on the correct device
    if models_loaded:
        try:
            # Check if omniparser has the required models
            _ = omniparser.som_model
            _ = omniparser.caption_model_processor
        except Exception:
            model_ok = False

    status = "ok" if models_loaded and cuda_ok and model_ok else "error"
    return {
        "status": status,
        "models_loaded": models_loaded,
        "cuda_ok": cuda_ok,
        "model_ok": model_ok,
        "device": device,
        "memory_info": {
            "cuda_memory_allocated": cuda_memory_allocated,
            "cuda_memory_reserved": cuda_memory_reserved
        }
    }


class ImageRequest(BaseModel):
    image: str
    
    @validator('image')
    def validate_image(cls, v):
        """Validate that the image string is a valid base64 encoded image."""
        if not v:
            raise ValueError('Image data cannot be empty')
        
        try:
            # Try to decode base64
            image_data = base64.b64decode(v)
            
            # Try to open as PIL Image to validate it's a valid image
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify the image
            
            return v
        except Exception as e:
            raise ValueError(f'Invalid image data: {str(e)}')


@app.post("/api/parse")
async def parse(request: ImageRequest):
    """Parse a base64 encoded image using OmniParser's image processing pipeline."""
    global omniparser
    
    # Check if models are loaded
    if omniparser is None:
        logger.error("OmniParser models not initialized")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable - models not loaded"
        )
    
    logger.info('Starting image parsing...')
    start = time.time()
    
    try:
        # Validate and decode the image
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
            image.load()  # Load the image data
        except Exception as e:
            logger.error(f"Failed to decode/load image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )
        
        # Check image dimensions and size
        width, height = image.size
        if width > 4096 or height > 4096:
            logger.warning(f"Large image detected: {width}x{height}")
        
        # Process the image
        try:
            dino_labled_img, parsed_content_list = omniparser.parse(request.image)
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error during parsing")
            cleanup_memory()
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable - GPU memory exhausted"
            )
        except Exception as e:
            logger.error(f"Error during image parsing: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error during image processing: {str(e)}"
            )
        
        # Validate the response
        if not isinstance(parsed_content_list, list):
            logger.error("Invalid response format from omniparser")
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from image processor"
            )
        
        latency = time.time() - start
        logger.info(f'Parsing completed successfully in {latency:.2f} seconds')
        
        # Clean up memory after processing
        cleanup_memory()
        
        return {
            "img": dino_labled_img, 
            "elements": parsed_content_list, 
            "latency": latency,
            "image_info": {
                "width": width,
                "height": height,
                "format": image.format
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Unexpected error in parse route: {str(e)}")
        cleanup_memory()
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )