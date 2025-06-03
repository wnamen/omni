import logging
import sys
import time
import os
import gc

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add OmniParser to system path
sys.path.insert(0, os.path.join(os.getcwd(), 'OmniParser'))

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for models
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = None
caption_model_processor = None

# Image processing configuration
MAX_IMAGE_SIZE = 1920  # Maximum width or height
MAX_TOTAL_PIXELS = 1920 * 1280  # Maximum total pixels (about 2.4MP)


def cleanup_memory():
    """Explicitly clean up memory and force garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def create_dtype_safe_model_wrapper(model, processor):
    """Create a wrapper that ensures consistent dtype handling for Florence-2 model."""
    class DTypeSafeModelWrapper:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor
            
        def generate(self, **kwargs):
            # Ensure pixel values are in the same dtype as the model, but preserve integer tensors
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    # Only convert continuous tensors like pixel_values to match model dtype
                    # Keep discrete tensors like input_ids and attention_mask as integers
                    if key in ['pixel_values'] and value.dtype.is_floating_point:
                        target_dtype = next(self.model.parameters()).dtype
                        if value.dtype != target_dtype:
                            logger.info(f"Converting {key} from {value.dtype} to {target_dtype}")
                            kwargs[key] = value.to(target_dtype)
                    elif key in ['input_ids', 'attention_mask'] and not value.dtype.is_floating_point:
                        # Ensure these remain as integers (typically Long)
                        if value.dtype != torch.long:
                            logger.info(f"Converting {key} to long (integer) type")
                            kwargs[key] = value.to(torch.long)
            
            return self.model.generate(**kwargs)
            
        def __getattr__(self, name):
            # Delegate other attributes to the original model
            return getattr(self.model, name)
    
    return DTypeSafeModelWrapper(model, processor)


def resize_image_if_needed(image: Image.Image) -> tuple[Image.Image, bool]:
    """
    Resize image if it's too large to prevent memory issues.
    Returns: (resized_image, was_resized)
    """
    original_size = image.size
    width, height = original_size
    total_pixels = width * height
    
    logger.info(f"Image size check: {width}x{height} = {total_pixels:,} pixels")
    logger.info(f"Limits: max_size={MAX_IMAGE_SIZE}, max_pixels={MAX_TOTAL_PIXELS:,}")
    
    # Check if resizing is needed
    needs_resize = (
        width > MAX_IMAGE_SIZE or 
        height > MAX_IMAGE_SIZE or 
        total_pixels > MAX_TOTAL_PIXELS
    )
    
    logger.info(f"Needs resize: {needs_resize} (width>{MAX_IMAGE_SIZE}: {width > MAX_IMAGE_SIZE}, height>{MAX_IMAGE_SIZE}: {height > MAX_IMAGE_SIZE}, pixels>{MAX_TOTAL_PIXELS}: {total_pixels > MAX_TOTAL_PIXELS})")
    
    if not needs_resize:
        logger.info("No resizing needed")
        return image, False
    
    # Calculate new size maintaining aspect ratio
    aspect_ratio = width / height
    
    # Method 1: Scale down based on max dimension
    if width > height:
        new_width = min(width, MAX_IMAGE_SIZE)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(height, MAX_IMAGE_SIZE)
        new_width = int(new_height * aspect_ratio)
    
    # Method 2: Further scale down if total pixels still too high
    new_total_pixels = new_width * new_height
    if new_total_pixels > MAX_TOTAL_PIXELS:
        scale_factor = (MAX_TOTAL_PIXELS / new_total_pixels) ** 0.5
        new_width = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)
    
    # Ensure minimum size
    new_width = max(new_width, 100)
    new_height = max(new_height, 100)
    
    logger.info(f"Resizing image from {original_size} to ({new_width}, {new_height})")
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image, True


# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global yolo_model, caption_model_processor
    
    try:
        logger.info(f"Initializing models on device: {device}")
        
        # Import OmniParser utilities
        from transformers import AutoProcessor, AutoModelForCausalLM
        from util.utils import get_yolo_model
        
        # Load YOLO model
        logger.info("Loading YOLO model...")
        model_path = os.path.join(os.getcwd(), 'weights/icon_detect/model.pt')
        if os.path.exists(model_path):
            yolo_model = get_yolo_model(model_path=model_path)
            yolo_model = yolo_model.float().to(device)
            logger.info(f"YOLO model loaded successfully on {device}")
        else:
            logger.error(f"YOLO model not found at path: {model_path}")
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        
        # Load Florence-2 caption model
        logger.info("Loading Florence-2 caption model...")
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        # Ensure model and all its parameters are in float32
        model = model.float()
        model = model.to(device)
        
        # Create wrapper to handle dtype issues
        wrapped_model = create_dtype_safe_model_wrapper(model, processor)
        caption_model_processor = {'model': wrapped_model, 'processor': processor}
        logger.info("Caption model loaded successfully")
        
        logger.info("All models initialized successfully")
        
        # Clean up after model loading
        cleanup_memory()
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        # Don't raise the exception to allow the server to start anyway
        # but log the error for debugging
        logger.exception("Model initialization failed")


@app.get("/api/health")
async def health_check():
    global yolo_model, caption_model_processor
    
    # Check if models are loaded
    models_loaded = yolo_model is not None and caption_model_processor is not None
    
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "device": device,
        "memory_info": {
            "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
    }


class ImageRequest(BaseModel):
    image: str


@app.post("/api/parse")
async def parse(request: ImageRequest):
    """Parse a base64 encoded image using OmniParser's image processing pipeline."""
    logger.info("🔥 PARSE ENDPOINT CALLED 🔥")
    
    # Variables to track for cleanup
    image = None
    image_data = None
    temp_path = None
    
    try:
        # Get the base64 image from request body
        base64_image = request.image
        
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64 image (minimize memory by not storing intermediate variables)
        import base64
        from io import BytesIO
        
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        
        # Clear base64 data immediately to free memory
        del base64_image
        
        logger.info(f"🖼️ Original image loaded: {image.size} pixels")
        
        # Resize image if it's too large to prevent memory issues
        original_size = image.size
        resized_image, was_resized = resize_image_if_needed(image)
        
        # If resized, replace original with resized version to save memory
        if was_resized:
            logger.info(f"Image resized from {original_size} to {resized_image.size} to prevent memory issues")
            image.close()  # Explicitly close original image
            image = resized_image
        
        # Store processed size before closing the image
        processed_size = image.size
        
        # Clear image_data now that we have the PIL image
        del image_data

        from util.utils import get_som_labeled_img, check_ocr_box
        
        # Save temporarily for processing (using a more unique filename)
        temp_path = f"temp_{int(time.time() * 1000)}_{os.getpid()}.png"
        image.save(temp_path)
        
        # Close the PIL image to free memory since we have the file
        image.close()
        
        BOX_TRESHOLD = 0.05
        box_overlay_ratio = max(original_size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        try:
            # Process with OCR
            start_time = time.time()
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                temp_path,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                use_paddleocr=True
            )
            text, ocr_bbox = ocr_bbox_rslt
            ocr_time = time.time() - start_time
            
            # Clear intermediate OCR results we don't need
            del ocr_bbox_rslt, is_goal_filtered
            
            # Process with OmniParser
            start_time = time.time()
            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                temp_path,
                yolo_model,
                BOX_TRESHOLD=BOX_TRESHOLD,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=0.7,
                scale_img=False,
                batch_size=64  # Reduced batch size to save memory
            )
            parsing_time = time.time() - start_time
            
            # Clear intermediate variables
            del text, ocr_bbox, label_coordinates
            
            # Convert to simple list of dicts instead of DataFrame to save memory
            elements = []
            for i, item in enumerate(parsed_content_list):
                if isinstance(item, dict):
                    element = item.copy()
                    element['ID'] = i
                    elements.append(element)
                else:
                    # Handle non-dict items
                    elements.append({'ID': i, 'content': str(item)})
            
            # Clear parsed_content_list immediately
            del parsed_content_list
            
            # Prepare response with metadata about resizing
            response = {
                "img": dino_labeled_img,  # Base64 encoded image
                "elements": elements,
                "metadata": {
                    "original_size": original_size,
                    "processed_size": processed_size,
                    "was_resized": was_resized
                },
                "timing": {
                    "ocr_time": ocr_time,
                    "parsing_time": parsing_time,
                    "total_time": ocr_time + parsing_time
                }
            }
            
            # Clear large variables before returning
            del elements, dino_labeled_img
            
            return response
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not remove temporary file {temp_path}: {cleanup_error}")
                
    except Exception as e:
        logger.exception(f"Error parsing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Explicit cleanup of remaining variables
        try:
            if image:
                image.close()
            if 'image_data' in locals():
                del image_data
        except:
            pass
        
        # Force garbage collection after each request
        cleanup_memory()
        logger.info("Memory cleanup completed")