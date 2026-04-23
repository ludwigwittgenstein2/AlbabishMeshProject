"""
Albabish Medical Mesh CV Backend (Flask)
=================================================

What this backend does
- ✅ Headless CPU-only mesh rendering (no pyglet, no OpenGL)
- ✅ Multi-view CV stats (4 views): Sobel/Canny edges, SIFT/ORB keypoints, KMeans segmentation, contours
- ✅ Optional base64 images per view (include_images=true)
- ✅ Optional ViT classification on a rendered view
- ✅ Optional LLaVA-Med report from a 2x2 montage of the 4 rendered views
- ✅ Optional MedGemma report from a 2x2 montage of the 4 rendered views via Ollama

How to run
  python app.py

Frontend contract
  POST /api/upload   (multipart/form-data: file=<.stl or .obj>)
  POST /api/analyze  (json: { "filename": "x.stl", "options": {...} })

Backend return conventions
- Always return: ok, filename, device, geometry, views, patterns, timing_ms, backend_report
- If ViT enabled: return vit {status, ...}
- If LLaVA enabled: return llava_med {status, text, message}
- If MedGemma enabled: return medgemma {status, text, message, model}
"""

import os
import time
import base64
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# Environment safety (macOS / OpenCV / Matplotlib)
# ------------------------------------------------------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "false"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import cv2
import numpy as np
import trimesh
import torch
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ------------------------------------------------------------
# Optional models
# ------------------------------------------------------------
VIT_AVAILABLE = False
VIT_IMPORT_ERROR = None
try:
    from vit_infer import vit_predict
    VIT_AVAILABLE = True
except Exception as e:
    VIT_AVAILABLE = False
    VIT_IMPORT_ERROR = str(e)

LLAVA_AVAILABLE = False
LLAVA_IMPORT_ERROR = None
try:
    from llava_med_infer import llava_med_analyze
    LLAVA_AVAILABLE = True
except Exception as e:
    LLAVA_AVAILABLE = False
    LLAVA_IMPORT_ERROR = str(e)

MEDGEMMA_AVAILABLE = False
MEDGEMMA_IMPORT_ERROR = None
try:
    from medgemma_infer import medgemma_analyze
    MEDGEMMA_AVAILABLE = True
except Exception as e:
    MEDGEMMA_AVAILABLE = False
    MEDGEMMA_IMPORT_ERROR = str(e)

# ------------------------------------------------------------
# Flask setup
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".stl", ".obj"}

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def now_ms() -> int:
    return int(time.time() * 1000)

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    amin = float(arr.min())
    amax = float(arr.max())
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - amin) / (amax - amin)
    return (out * 255.0).clip(0, 255).astype(np.uint8)

def img_to_base64_png(img_rgb_or_gray: np.ndarray) -> str:
    """
    Accepts RGB uint8 HxWx3 or gray HxW.
    Returns base64 PNG payload (no data: prefix).
    """
    if img_rgb_or_gray is None:
        return ""
    if img_rgb_or_gray.ndim == 2:
        enc = img_rgb_or_gray
    else:
        enc = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", enc)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def make_montage_rgb(views_rgb: List[np.ndarray], grid=(2, 2), pad: int = 4) -> np.ndarray:
    """
    Combine 4 RGB images into a single 2x2 montage.
    Each view must be uint8 HxWx3 and same size.
    """
    rows, cols = grid
    if rows * cols != len(views_rgb):
        raise ValueError("grid must match number of views")
    h, w = views_rgb[0].shape[:2]
    H = rows * h + (rows - 1) * pad
    W = cols * w + (cols - 1) * pad
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0 = r * (h + pad)
            x0 = c * (w + pad)
            canvas[y0:y0 + h, x0:x0 + w] = views_rgb[idx]
            idx += 1
    return canvas

# ------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------
class MedicalMeshAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")

    # ---------------------------
    # Mesh loading + geometry
    # ---------------------------
    def load_mesh(self, filepath: Path) -> Optional[trimesh.Trimesh]:
        try:
            mesh = trimesh.load_mesh(str(filepath), force="mesh")
            if mesh is None:
                return None

            if isinstance(mesh, trimesh.Scene):
                parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(parts) if parts else None

            return mesh
        except Exception as e:
            print(f"❌ Mesh load error: {e}")
            return None

    def analyze_mesh_geometry(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        try:
            vol = float(mesh.volume) if mesh.is_watertight else 0.0
        except Exception:
            vol = 0.0

        try:
            area = float(mesh.area)
        except Exception:
            area = 0.0

        try:
            com = mesh.center_mass.tolist()
        except Exception:
            com = [0.0, 0.0, 0.0]

        try:
            bounds = mesh.bounds.tolist()
        except Exception:
            bounds = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        try:
            euler_number = int(mesh.euler_number)
        except Exception:
            euler_number = 0

        return {
            "vertices": safe_int(len(mesh.vertices)),
            "faces": safe_int(len(mesh.faces)),
            "volume": safe_float(vol),
            "surface_area": safe_float(area),
            "center_of_mass": com,
            "bounds": bounds,
            "is_watertight": bool(mesh.is_watertight),
            "euler_number": euler_number,
        }

    # ---------------------------
    # Headless CPU renderer (wireframe, 2D projection)
    # ---------------------------
    def render_mesh_views(self, mesh: trimesh.Trimesh, image_size: int = 512) -> List[np.ndarray]:
        angles = [
            (0, 0, 0),      # front
            (90, 0, 0),     # top
            (0, 90, 0),     # side
            (45, 45, 0),    # iso
        ]

        out: List[np.ndarray] = []
        for (ax, ay, az) in angles:
            m = mesh.copy()
            rot = trimesh.transformations.euler_matrix(
                np.deg2rad(ax), np.deg2rad(ay), np.deg2rad(az)
            )
            m.apply_transform(rot)

            verts = np.asarray(m.vertices, dtype=np.float32)
            if verts.shape[0] == 0:
                out.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
                continue

            xy = verts[:, :2]
            min_xy = xy.min(axis=0)
            max_xy = xy.max(axis=0)
            span = max_xy - min_xy
            span_max = float(max(span.max(), 1e-9))

            scale = (image_size * 0.8) / span_max
            center = (min_xy + max_xy) / 2.0

            pts = (xy - center) * scale + (image_size / 2.0)
            pts = np.clip(pts, 0, image_size - 1).astype(np.int32)

            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            edges = np.asarray(m.edges_unique, dtype=np.int32)

            if edges.shape[0] > 200_000:
                step = (edges.shape[0] // 200_000) + 1
                edges = edges[::step]

            for e0, e1 in edges:
                p0 = tuple(pts[e0])
                p1 = tuple(pts[e1])
                cv2.line(img, p0, p1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

            out.append(img)

        return out

    # ---------------------------
    # Image conversion
    # ---------------------------
    def to_gray(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image[..., 0].astype(np.uint8)

    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.astype(np.uint8)
        return image[..., :3].astype(np.uint8)

    # ---------------------------
    # CV modules
    # ---------------------------
    def edge_detection(self, image: np.ndarray) -> Dict[str, Any]:
        gray = self.to_gray(image)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)

        mag_u8 = normalize_to_uint8(mag)
        edges = cv2.Canny(gray, 50, 150)

        return {
            "edge_density": safe_float(np.sum(edges > 0) / max(edges.size, 1)),
            "sobel_magnitude_img": mag_u8,
            "canny_edges_img": edges,
            "gradient_x_img": normalize_to_uint8(np.abs(sobelx)),
            "gradient_y_img": normalize_to_uint8(np.abs(sobely)),
        }

    def detect_features(self, image: np.ndarray) -> Dict[str, Any]:
        gray = self.to_gray(image)
        rgb = self.to_rgb(image)

        try:
            sift = cv2.SIFT_create()
            kps, desc = sift.detectAndCompute(gray, None)
            kp_img = cv2.drawKeypoints(
                rgb, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            return {
                "method": "SIFT",
                "keypoints": safe_int(len(kps)),
                "descriptors_shape": desc.shape if desc is not None else (0, 0),
                "keypoints_img": kp_img,
            }
        except Exception:
            pass

        try:
            orb = cv2.ORB_create(nfeatures=1500)
            kps, desc = orb.detectAndCompute(gray, None)
            kp_img = cv2.drawKeypoints(rgb, kps, None, color=(255, 0, 0))
            return {
                "method": "ORB",
                "keypoints": safe_int(len(kps)),
                "descriptors_shape": desc.shape if desc is not None else (0, 0),
                "keypoints_img": kp_img,
            }
        except Exception as e:
            return {
                "method": "NONE",
                "keypoints": 0,
                "descriptors_shape": (0, 0),
                "error": str(e),
                "keypoints_img": rgb,
            }

    def segment_kmeans(self, image: np.ndarray, k: int = 5) -> Dict[str, Any]:
        rgb = self.to_rgb(image)
        pixels = rgb.reshape((-1, 3)).astype(np.float32)

        k = int(max(2, min(k, 10)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        try:
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            centers_u8 = np.uint8(centers)
            segmented = centers_u8[labels.flatten()].reshape(rgb.shape)
            return {
                "num_clusters": k,
                "cluster_centers": centers_u8.tolist(),
                "segmented_img": segmented,
            }
        except Exception as e:
            return {
                "num_clusters": k,
                "cluster_centers": [],
                "segmented_img": rgb,
                "error": str(e),
            }

    def detect_contours(self, image: np.ndarray) -> Dict[str, Any]:
        gray = self.to_gray(image)
        rgb = self.to_rgb(image)

        try:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        except Exception:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_img = rgb.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

        stats = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 50:
                continue
            perimeter = float(cv2.arcLength(cnt, True))
            circ = float(4 * np.pi * area / (perimeter * perimeter)) if perimeter > 1e-9 else 0.0
            stats.append({"area": area, "perimeter": perimeter, "circularity": circ})
        stats.sort(key=lambda x: x["area"], reverse=True)

        return {
            "num_contours": safe_int(len(contours)),
            "top_contours": stats[:10],
            "contour_img": contour_img,
        }

    # ---------------------------
    # Report block
    # ---------------------------
    def build_backend_report(self, results: Dict[str, Any]) -> str:
        geom = results.get("geometry", {}) or {}
        pat = results.get("patterns", {}) or {}
        vit = results.get("vit", {}) or {}
        llava = results.get("llava_med", {}) or {}
        medgemma = results.get("medgemma", {}) or {}

        lines = []
        lines.append("3D MEDICAL MORPHOLOGY REVIEW REPORT")
        lines.append("==================================================")
        lines.append(f"File: {results.get('filename', 'unknown')}")
        lines.append(f"Processing status: {'OK' if results.get('ok') else 'ERROR'}")
        lines.append(f"Compute device: {results.get('device', 'unknown')}")
        lines.append("")

        lines.append("1. Mesh Reconstruction Summary")
        lines.append("--------------------------------------------------")
        lines.append(f"Vertices: {geom.get('vertices', '—')}")
        lines.append(f"Faces: {geom.get('faces', '—')}")
        lines.append(f"Watertight mesh: {geom.get('is_watertight', '—')}")
        lines.append(f"Volume: {geom.get('volume', '—')}")
        lines.append(f"Surface area: {geom.get('surface_area', '—')}")
        lines.append(f"Euler number: {geom.get('euler_number', '—')}")
        lines.append("")

        lines.append("2. Geometric Morphology Summary")
        lines.append("--------------------------------------------------")
        lines.append(f"Number of rendered views: {pat.get('num_views', '—')}")
        lines.append(f"Aggregate feature detections (SIFT/ORB): {pat.get('total_features', '—')}")
        lines.append(f"Average edge density across views: {pat.get('avg_edge_density', '—')}")
        lines.append(f"Aggregate contour count: {pat.get('total_contours', '—')}")
        lines.append("")

        lines.append("3. AI-Assisted Morphology Interpretation")
        lines.append("--------------------------------------------------")

        if medgemma.get("status") == "ok":
            lines.append("3.1 MedGemma Structured Morphology Review")
            lines.append(f"Model: {medgemma.get('model', '—')}")
            lines.append("")
            txt = (medgemma.get("text") or "").strip()
            if txt:
                lines.append(txt)
            else:
                lines.append("No structured morphology text was generated.")
            lines.append("")
        elif medgemma.get("status") == "error":
            lines.append("3.1 MedGemma Structured Morphology Review")
            lines.append("Status: error")
            lines.append(f"Message: {medgemma.get('message', 'Unknown error')}")
            lines.append("")
        else:
            lines.append("3.1 MedGemma Structured Morphology Review")
            lines.append("Status: skipped")
            lines.append("")

        if llava.get("status") == "ok":
            lines.append("3.2 LLaVA-Med Supplemental Interpretation")
            lines.append("")
            txt = (llava.get("text") or "").strip()
            if txt:
                lines.append(txt)
            else:
                lines.append("No supplemental interpretation text was generated.")
            lines.append("")
        elif llava.get("status") == "error":
            lines.append("3.2 LLaVA-Med Supplemental Interpretation")
            lines.append("Status: error")
            lines.append(f"Message: {llava.get('message', 'Unknown error')}")
            lines.append("")
        else:
            lines.append("3.2 LLaVA-Med Supplemental Interpretation")
            lines.append("Status: skipped")
            lines.append("")

        lines.append("4. Optional Visual Classification")
        lines.append("--------------------------------------------------")
        lines.append(f"ViT status: {vit.get('status', 'skipped')}")
        if vit.get("status") == "ok":
            lines.append(f"Predicted label: {vit.get('label', '—')}")
            lines.append(f"Predicted class ID: {vit.get('predicted_class', '—')}")
            lines.append(f"Confidence: {vit.get('confidence', '—')}")
        elif vit.get("message"):
            lines.append(f"Message: {vit.get('message')}")
        lines.append("")

        lines.append("5. Interpretive Limitations")
        lines.append("--------------------------------------------------")
        lines.append("• This report is based on rendered surface mesh geometry, not raw imaging voxels.")
        lines.append("• Internal tissue composition, attenuation, enhancement, signal characteristics, and histology are not directly available from surface mesh geometry.")
        lines.append("• Morphological observations may reflect segmentation quality, reconstruction artifacts, or rendering limitations.")
        lines.append("• AI-generated anatomical interpretation should be treated as supportive review, not diagnosis.")
        lines.append("")

        return "\n".join(lines)

    # ---------------------------
    # Full pipeline
    # ---------------------------
    def full_analysis(
        self,
        filepath: Path,
        include_images: bool = False,
        include_vit: bool = False,
        include_llava_med: bool = False,
        include_medgemma: bool = False,
        llava_prompt: Optional[str] = None,
        medgemma_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = now_ms()
        results: Dict[str, Any] = {
            "filename": filepath.name,
            "ok": False,
            "device": str(self.device),
            "geometry": {},
            "views": [],
            "patterns": {},
            "timing_ms": {},
            "vit": {"status": "skipped"},
            "llava_med": {"status": "skipped", "text": "", "message": ""},
            "medgemma": {"status": "skipped", "text": "", "message": "", "model": ""},
        }

        mesh = self.load_mesh(filepath)
        if mesh is None:
            results["error"] = "Failed to load mesh"
            return results

        results["geometry"] = self.analyze_mesh_geometry(mesh)

        t_r0 = now_ms()
        views = self.render_mesh_views(mesh, image_size=512)
        results["timing_ms"]["render_views"] = now_ms() - t_r0

        # ViT
        if include_vit:
            if not VIT_AVAILABLE:
                results["vit"] = {
                    "status": "error",
                    "message": f"vit_infer import failed: {VIT_IMPORT_ERROR}"
                }
            else:
                try:
                    v = self.to_rgb(views[3]).astype(np.uint8)
                    pil = Image.fromarray(v).convert("RGB")
                    results["vit"] = vit_predict(pil, topk=5)
                except Exception as e:
                    results["vit"] = {"status": "error", "message": str(e)}

        # LLaVA-Med
        if include_llava_med:
            if not LLAVA_AVAILABLE:
                results["llava_med"] = {
                    "status": "error",
                    "text": "",
                    "message": f"llava_med_infer import failed: {LLAVA_IMPORT_ERROR}",
                }
            else:
                try:
                    rgb_views = [self.to_rgb(v).astype(np.uint8) for v in views]
                    montage = make_montage_rgb(rgb_views, grid=(2, 2), pad=4)
                    pil = Image.fromarray(montage).convert("RGB")

                    prompt = llava_prompt or (
                        "You are a medical vision-language analysis expert.\n"
                        "Input: a 2x2 montage showing four perspectives of a 3D anatomical mesh rendered as a green wireframe on a black background.\n\n"
                        "Task:\n"
                        "1) Identify and describe consistent visual or geometric patterns (symmetry, curvature, branching, segmentation, or surface contour).\n"
                        "2) Interpret these patterns in an anatomy-oriented way using knowledge from medical morphology literature.\n"
                        "3) Provide anatomical hypotheses with uncertainty — do not diagnose.\n"
                        "4) State what additional data would improve analysis.\n\n"
                        "Output format:\n"
                        "Detected geometric patterns:\n"
                        "- ...\n"
                        "Anatomical interpretation (tentative):\n"
                        "- ...\n"
                        "Knowledge basis:\n"
                        "- ...\n"
                        "Limitations and missing data:\n"
                        "- ...\n"
                    )

                    results["llava_med"] = llava_med_analyze(pil, prompt)
                    results["llava_med"]["text"] = results["llava_med"].get("text") or ""
                    results["llava_med"]["message"] = results["llava_med"].get("message") or ""
                except Exception as e:
                    results["llava_med"] = {"status": "error", "text": "", "message": str(e)}

        # MedGemma
        if include_medgemma:
            if not MEDGEMMA_AVAILABLE:
                results["medgemma"] = {
                    "status": "error",
                    "text": "",
                    "message": f"medgemma_infer import failed: {MEDGEMMA_IMPORT_ERROR}",
                    "model": "",
                }
            else:
                try:
                    rgb_views = [self.to_rgb(v).astype(np.uint8) for v in views]
                    montage = make_montage_rgb(rgb_views, grid=(2, 2), pad=4)
                    pil = Image.fromarray(montage).convert("RGB")

                    prompt = medgemma_prompt or (
                        "You are an expert assistant for 3D medical morphology review.\n\n"
                        "You are given a rendered montage of a 3D anatomical mesh reconstructed from medical segmentation.\n"
                        "Your role is to identify visible morphological patterns, possible structural anomalies, "
                        "and anatomically meaningful features using medical terminology.\n\n"
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
                        "- Clearly state uncertainty when anatomy or abnormality is not confidently identifiable.\n"
                        "- Write in concise report style.\n"
                    )

                    results["medgemma"] = medgemma_analyze(pil, prompt=prompt)
                    results["medgemma"]["text"] = results["medgemma"].get("text") or ""
                    results["medgemma"]["message"] = results["medgemma"].get("message") or ""
                    results["medgemma"]["model"] = results["medgemma"].get("model") or ""
                except Exception as e:
                    results["medgemma"] = {
                        "status": "error",
                        "text": "",
                        "message": str(e),
                        "model": "",
                    }

        # Classic CV per view
        total_features = 0
        edge_densities: List[float] = []
        total_contours = 0

        for idx, view_rgb in enumerate(views):
            view_rgb = self.to_rgb(view_rgb)

            edges = self.edge_detection(view_rgb)
            feats = self.detect_features(view_rgb)
            seg = self.segment_kmeans(view_rgb, k=5)
            cont = self.detect_contours(view_rgb)

            edge_densities.append(edges["edge_density"])
            total_features += safe_int(feats.get("keypoints", 0))
            total_contours += safe_int(cont.get("num_contours", 0))

            view_result: Dict[str, Any] = {
                "view_index": idx,
                "edge_detection": {"edge_density": edges["edge_density"]},
                "features": {
                    "method": feats.get("method", "UNKNOWN"),
                    "keypoints_detected": feats.get("keypoints", 0),
                    "descriptor_dimensions": feats.get("descriptors_shape", (0, 0)),
                },
                "segmentation": {
                    "clusters": seg.get("num_clusters", 0),
                    "cluster_centers": seg.get("cluster_centers", []),
                },
                "contours": {
                    "num_contours": cont.get("num_contours", 0),
                    "top_contours": cont.get("top_contours", []),
                },
            }

            if include_images:
                view_result["images_base64"] = {
                    "rendered_view": img_to_base64_png(view_rgb),
                    "sobel_magnitude": img_to_base64_png(edges["sobel_magnitude_img"]),
                    "canny_edges": img_to_base64_png(edges["canny_edges_img"]),
                    "grad_x": img_to_base64_png(edges["gradient_x_img"]),
                    "grad_y": img_to_base64_png(edges["gradient_y_img"]),
                    "keypoints": img_to_base64_png(feats.get("keypoints_img")),
                    "segmented": img_to_base64_png(seg.get("segmented_img")),
                    "contours": img_to_base64_png(cont.get("contour_img")),
                }

            results["views"].append(view_result)

        results["patterns"] = {
            "total_features": safe_int(total_features),
            "avg_edge_density": safe_float(float(np.mean(edge_densities)) if edge_densities else 0.0),
            "total_contours": safe_int(total_contours),
            "num_views": safe_int(len(views)),
        }

        results["timing_ms"]["total"] = now_ms() - t0
        results["ok"] = True
        results["backend_report"] = self.build_backend_report(results)
        return results


# ------------------------------------------------------------
# Init analyzer
# ------------------------------------------------------------
analyzer = MedicalMeshAnalyzer()

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(analyzer.device),
        "opencv_version": cv2.__version__,
        "vit_available": bool(VIT_AVAILABLE),
        "llava_med_available": bool(LLAVA_AVAILABLE),
        "medgemma_available": bool(MEDGEMMA_AVAILABLE),
        "vit_import_error": VIT_IMPORT_ERROR,
        "llava_import_error": LLAVA_IMPORT_ERROR,
        "medgemma_import_error": MEDGEMMA_IMPORT_ERROR,
    })

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(f.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Only .stl and .obj files are allowed"}), 400

    filepath = UPLOAD_FOLDER / filename
    f.save(filepath)

    return jsonify({
        "message": "File uploaded successfully",
        "filename": filename,
        "path": str(filepath),
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        filename = data.get("filename")
        options = data.get("options") or {}

        include_images = bool(options.get("include_images", data.get("include_images", False)))
        include_vit = bool(options.get("include_vit", data.get("include_vit", False)))
        include_llava_med = bool(options.get("include_llava_med", data.get("include_llava_med", True)))
        include_medgemma = bool(options.get("include_medgemma", data.get("include_medgemma", False)))

        llava_prompt = options.get("llava_prompt", None)
        medgemma_prompt = options.get("medgemma_prompt", None)

        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        filename = secure_filename(filename)
        filepath = UPLOAD_FOLDER / filename
        if not filepath.exists():
            return jsonify({"error": f"File not found: {filename}"}), 404

        results = analyzer.full_analysis(
            filepath,
            include_images=include_images,
            include_vit=include_vit,
            include_llava_med=include_llava_med,
            include_medgemma=include_medgemma,
            llava_prompt=llava_prompt,
            medgemma_prompt=medgemma_prompt,
        )

        return jsonify(results), (200 if results.get("ok") else 500)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except Exception:
        pass

    print("🚀 Starting Albabish Medical Mesh CV Analysis Server...")
    print(f"   Device: {analyzer.device}")
    print("   Headless CPU rendering: ✅ enabled (no pyglet/OpenGL)")
    app.run(host="0.0.0.0", port=8000, threaded=True, debug=False)