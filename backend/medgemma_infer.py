# medgemma_infer.py
import base64
import io
import os
from typing import Any, Dict, Optional

import requests
from PIL import Image

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL", "medgemma:latest")


def _pil_to_base64_jpeg(pil_image: Image.Image, quality: int = 95) -> str:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def medgemma_analyze(
    pil_image: Image.Image,
    prompt: Optional[str] = None,
    timeout_sec: int = 180,
) -> Dict[str, Any]:
    try:
        image_b64 = _pil_to_base64_jpeg(pil_image)

        user_prompt = prompt or (   "You are an expert assistant for 3D medical morphology review.\n\n"
                                    "You are given a rendered view of a 3D anatomical mesh reconstructed from medical segmentation.\n"
                                    "Your role is to identify visible morphological patterns, possible structural anomalies, and anatomically meaningful features using medical terminology.\n\n"
                                    "Assess, when visible:\n"
                                    "- global shape and orientation\n"
                                    "- symmetry / asymmetry\n"
                                    "- contour regularity\n"
                                    "- surface smoothness vs. irregularity\n"
                                    "- lobulation\n"
                                    "- branching architecture\n"
                                    "- focal bulging, indentation, stenosis, dilation, tapering, or distortion\n"
                                    "- continuity of boundary\n"
                                    "- anatomical plausibility of the segmented structure\n\n"
                                    "Provide the response in the following format:\n"
                                    "Structure overview:\n"
                                    "- ...\n"
                                    "Morphological observations:\n"
                                    "- ...\n"
                                    "Potential anatomical significance:\n"
                                    "- ...\n"
                                    "Possible anomalous or abnormal geometric features:\n"
                                    "- ...\n"
                                    "Limitations:\n"
                                    "- ...\n\n"
                                    "Constraints:\n"
                                    "- Use medical terminology where justified.\n"
                                    "- Do not diagnose disease.\n"
                                    "- Do not infer histology, tissue density, or radiology findings from mesh geometry alone.\n"
                                    "- Clearly state uncertainty when anatomy or abnormality is not confidently identifiable.\n")

        payload = {
            "model": MEDGEMMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "")

        return {
            "status": "ok",
            "model": MEDGEMMA_MODEL,
            "text": text,
            "message": "",
        }

    except Exception as e:
        return {
            "status": "error",
            "model": MEDGEMMA_MODEL,
            "text": "",
            "message": str(e),
        }