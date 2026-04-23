# vit_infer.py
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Pick a standard ViT. (ImageNet labels; not “medical” semantics)
VIT_MODEL_ID = "google/vit-base-patch16-224"

_device = None
_processor = None
_model = None

def _lazy_load():
    global _device, _processor, _model
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID, use_fast=True)
    _model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_ID)
    _model.to(_device)
    _model.eval()

@torch.inference_mode()
def vit_predict(pil_image: Image.Image, topk: int = 5):
    """
    Returns:
      {
        status: "ok",
        device: "...",
        predicted_class: int,
        label: str,
        confidence: float,
        topk: [{class_id,label,prob}, ...]
      }
    """
    _lazy_load()

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    inputs = _processor(images=pil_image, return_tensors="pt").to(_device)
    logits = _model(**inputs).logits  # (1, num_classes)
    probs = torch.softmax(logits, dim=-1)[0]

    conf, idx = torch.max(probs, dim=0)
    predicted_class = int(idx.item())
    confidence = float(conf.item())

    # label mapping from model config
    id2label = _model.config.id2label
    label = id2label.get(predicted_class, str(predicted_class))

    # topk
    topk = min(topk, probs.numel())
    top_probs, top_ids = torch.topk(probs, k=topk)
    top = []
    for p, cid in zip(top_probs.tolist(), top_ids.tolist()):
        top.append({
            "class_id": int(cid),
            "label": id2label.get(int(cid), str(int(cid))),
            "prob": float(p),
        })

    return {
        "status": "ok",
        "device": _device,
        "predicted_class": predicted_class,
        "label": label,
        "confidence": confidence,
        "topk": top,
    }
