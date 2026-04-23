"""
LLaVA-Med inference module - Fixed version
Uses proven HuggingFace models that have proper processor configs
"""
import torch
from PIL import Image

# Use proven models with complete processor configurations
# microsoft/llava-med-v1.5-mistral-7b DOES NOT WORK - missing processor files
MODEL_OPTIONS = [
    "llava-hf/llava-1.5-7b-hf"           # Primary choice - most stable
]

_device = None
_processor = None
_model = None
_model_id_used = None

def _pick_device():
    """Select best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _lazy_load():
    """Lazy load model with fallback to alternative models"""
    global _device, _processor, _model, _model_id_used
    
    if _model is not None:
        return

    _device = _pick_device()
    dtype = torch.float16 if _device.type in ("cuda", "mps") else torch.float32

    # Try each model until one works
    last_error = None
    for model_id in MODEL_OPTIONS:
        try:
            print(f"🔄 Attempting to load: {model_id}")
            
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            _processor = AutoProcessor.from_pretrained(model_id)
            _model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(_device)
            
            _model.eval()
            _model_id_used = model_id
            print(f"✅ Successfully loaded: {model_id}")
            return
            
        except Exception as e:
            last_error = e
            print(f"⚠️ Failed to load {model_id}: {str(e)[:100]}")
            continue
    
    # If all models failed, raise the last error
    raise RuntimeError(
        f"Failed to load any LLaVA model. Last error: {last_error}\n"
        f"Tried models: {MODEL_OPTIONS}"
    )

@torch.inference_mode()
def llava_med_analyze(pil_image: Image.Image, prompt: str, max_new_tokens: int = 300):
    """
    Analyze medical image using LLaVA model
    
    Args:
        pil_image: PIL Image (the 2x2 montage from app.py)
        prompt: Text prompt for analysis
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        dict with: status, device, text, message, model_used
    """
    try:
        _lazy_load()
    except Exception as e:
        return {
            "status": "error",
            "text": "",
            "message": f"Model loading failed: {str(e)}",
            "device": "none"
        }

    try:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Format conversation for LLaVA
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ],
        }]

        # Apply chat template and prepare inputs
        text = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(images=pil_image, text=text, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        # Generate response
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        # Decode output
        out_text = _processor.decode(out_ids[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "ASSISTANT:" in out_text:
            out_text = out_text.split("ASSISTANT:")[-1].strip()
        elif "assistant\n" in out_text.lower():
            out_text = out_text.lower().split("assistant\n")[-1].strip()
        
        return {
            "status": "ok",
            "device": str(_device),
            "text": out_text,
            "message": f"Generated using {_model_id_used}",
            "model_used": _model_id_used
        }
        
    except Exception as e:
        return {
            "status": "error",
            "text": "",
            "message": f"Inference failed: {str(e)}",
            "device": str(_device) if _device else "none"
        }


if __name__ == "__main__":
    # Simple test
    print("🧪 Testing LLaVA inference...")
    test_img = Image.new('RGB', (512, 512), color=(0, 128, 0))
    result = llava_med_analyze(test_img, "Describe this image briefly.", max_new_tokens=50)
    
    print(f"\nStatus: {result['status']}")
    print(f"Device: {result['device']}")
    if result.get('model_used'):
        print(f"Model: {result['model_used']}")
    if result['text']:
        print(f"Output: {result['text'][:200]}")
    if result.get('message'):
        print(f"Message: {result['message']}")