import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Upload,
  Play,
  Download,
  Eye,
  Layers,
  AlertCircle,
  CheckCircle,
  Server,
  FileText,
  Image as ImageIcon,
  Brain,
} from "lucide-react";

/**
 * MedicalMeshCVAnalyzer.jsx
 * ----------------------------------------------------
 * STL / OBJ → 2D wireframe render → client CV (Sobel/Edges/Harris/Mask/KMeans)
 * Optional Backend:
 *   - Geometry + Multi-view CV stats
 *   - ViT classification
 *   - LLaVA-Med report (text)
 *   - MedGemma report (text)
 */

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

// ---------- Small helpers ----------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const nowISO = () => new Date().toISOString();

function formatNumber(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (typeof n === "number") return n.toFixed(digits);
  return String(n);
}

function coerceText(x) {
  if (x === null || x === undefined) return "";
  if (Array.isArray(x)) return x.map((v) => String(v)).join("\n");
  return String(x);
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadJSON(filename, obj) {
  const text = JSON.stringify(obj, null, 2);
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function get2D(canvas, willReadFrequently = false) {
  const ctx = canvas.getContext(
    "2d",
    willReadFrequently ? { willReadFrequently: true } : undefined
  );
  if (!ctx) throw new Error("Canvas 2D context is not available.");
  return ctx;
}

// ---------- STL parsing ----------
async function parseSTL(file) {
  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);

  const isBinary =
    buffer.byteLength > 84 && view.getUint32(80, true) * 50 + 84 === buffer.byteLength;

  const vertices = [];
  const faces = [];

  if (isBinary) {
    const numFaces = view.getUint32(80, true);
    let offset = 84;

    for (let i = 0; i < numFaces; i++) {
      offset += 12;

      const v1 = [
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ];
      offset += 12;

      const v2 = [
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ];
      offset += 12;

      const v3 = [
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ];
      offset += 12;

      offset += 2;

      const idx = vertices.length;
      vertices.push(v1, v2, v3);
      faces.push([idx, idx + 1, idx + 2]);
    }

    return { vertices, faces, format: "binary" };
  }

  const text = new TextDecoder().decode(buffer);
  const lines = text.split("\n");
  let tempVerts = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("vertex")) {
      const parts = trimmed.split(/\s+/);
      if (parts.length >= 4) {
        const x = parseFloat(parts[1]);
        const y = parseFloat(parts[2]);
        const z = parseFloat(parts[3]);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          tempVerts.push([x, y, z]);
          if (tempVerts.length === 3) {
            const idx = vertices.length;
            vertices.push(tempVerts[0], tempVerts[1], tempVerts[2]);
            faces.push([idx, idx + 1, idx + 2]);
            tempVerts = [];
          }
        }
      }
    }
  }

  return { vertices, faces, format: "ascii" };
}

// ---------- OBJ parsing ----------
async function parseOBJ(file) {
  const text = await file.text();
  const lines = text.split("\n");

  const vertices = [];
  const faces = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    if (line.startsWith("v ")) {
      const parts = line.split(/\s+/);
      if (parts.length >= 4) {
        const x = parseFloat(parts[1]);
        const y = parseFloat(parts[2]);
        const z = parseFloat(parts[3]);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          vertices.push([x, y, z]);
        }
      }
    } else if (line.startsWith("f ")) {
      const parts = line.split(/\s+/).slice(1);

      const faceIndices = parts
        .map((p) => {
          const idx = parseInt(p.split("/")[0], 10); // supports v, v/vt, v/vt/vn, v//vn
          return Number.isFinite(idx) ? idx - 1 : null; // OBJ is 1-based
        })
        .filter((idx) => idx !== null && idx >= 0);

      if (faceIndices.length >= 3) {
        // triangulate polygon fan-style
        for (let i = 1; i < faceIndices.length - 1; i++) {
          faces.push([faceIndices[0], faceIndices[i], faceIndices[i + 1]]);
        }
      }
    }
  }

  return { vertices, faces, format: "obj" };
}

// ---------- Canvas rendering ----------
function renderMeshToCanvas(vertices, faces, canvas, opts = {}) {
  const {
    maxFacesToDraw = 120_000,
    strokeColor = "#00ff00",
    bgColor = "#000000",
    lineWidth = 1,
  } = opts;

  const ctx = get2D(canvas, true);
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, w, h);

  if (!vertices?.length || !faces?.length) return null;

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];
    minX = Math.min(minX, v[0]);
    maxX = Math.max(maxX, v[0]);
    minY = Math.min(minY, v[1]);
    maxY = Math.max(maxY, v[1]);
  }

  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const scale = (Math.min(w, h) * 0.85) / Math.max(spanX, spanY);

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const project = (v) => {
    const x = (v[0] - centerX) * scale + w / 2;
    const y = (v[1] - centerY) * scale + h / 2;
    return [x, y];
  };

  let drawFaces = faces;
  if (faces.length > maxFacesToDraw) {
    const step = Math.ceil(faces.length / maxFacesToDraw);
    drawFaces = faces.filter((_, idx) => idx % step === 0);
  }

  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = lineWidth;

  for (let i = 0; i < drawFaces.length; i++) {
    const f = drawFaces[i];
    const p1 = project(vertices[f[0]]);
    const p2 = project(vertices[f[1]]);
    const p3 = project(vertices[f[2]]);

    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.lineTo(p3[0], p3[1]);
    ctx.closePath();
    ctx.stroke();
  }

  return canvas.toDataURL("image/png");
}

// ---------- Image processing ----------
function imageDataToGrayscaleFloat(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);

  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return { gray, width, height };
}

function convolveSobel(gray, width, height) {
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sumX = 0;
      let sumY = 0;
      let idxK = 0;

      for (let yy = -1; yy <= 1; yy++) {
        for (let xx = -1; xx <= 1; xx++) {
          const v = gray[(y + yy) * width + (x + xx)];
          sumX += v * kx[idxK];
          sumY += v * ky[idxK];
          idxK++;
        }
      }
      const idx = y * width + x;
      gx[idx] = sumX;
      gy[idx] = sumY;
    }
  }
  return { gx, gy };
}

function sobelMagnitudeImageData(imageData) {
  const { width, height } = imageData;
  const { gray } = imageDataToGrayscaleFloat(imageData);
  const { gx, gy } = convolveSobel(gray, width, height);

  let maxMag = 1e-9;
  const mag = new Float32Array(width * height);
  for (let i = 0; i < mag.length; i++) {
    const m = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    mag[i] = m;
    if (m > maxMag) maxMag = m;
  }

  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const v = clamp((mag[i] / maxMag) * 255, 0, 255);
    out[i * 4 + 0] = v;
    out[i * 4 + 1] = v;
    out[i * 4 + 2] = v;
    out[i * 4 + 3] = 255;
  }

  return new ImageData(out, width, height);
}

function edgeMapFromSobel(imageData, threshold = 70) {
  const sobel = sobelMagnitudeImageData(imageData);
  const { data, width, height } = sobel;
  const out = new Uint8ClampedArray(data.length);

  let edgeCount = 0;
  for (let i = 0; i < width * height; i++) {
    const v = data[i * 4];
    const isEdge = v >= threshold;
    const px = isEdge ? 255 : 0;
    out[i * 4 + 0] = px;
    out[i * 4 + 1] = px;
    out[i * 4 + 2] = px;
    out[i * 4 + 3] = 255;
    if (isEdge) edgeCount++;
  }

  const edgeDensity = edgeCount / (width * height);
  return { edgeImageData: new ImageData(out, width, height), edgeDensity };
}

function foregroundSegmentation(imageData, threshold = 15) {
  const { width, height } = imageData;
  const { gray } = imageDataToGrayscaleFloat(imageData);
  const out = new Uint8ClampedArray(width * height * 4);

  let fgCount = 0;
  for (let i = 0; i < width * height; i++) {
    const g = gray[i];
    const fg = g >= threshold;
    if (fg) fgCount++;

    out[i * 4 + 0] = 0;
    out[i * 4 + 1] = fg ? 255 : 0;
    out[i * 4 + 2] = fg ? 255 : 0;
    out[i * 4 + 3] = 255;
  }

  return {
    mask: new ImageData(out, width, height),
    foregroundRatio: fgCount / (width * height),
  };
}

function boxBlur(src, width, height, radius = 1) {
  const dst = new Float32Array(width * height);
  const k = (2 * radius + 1) ** 2;

  for (let y = radius; y < height - radius; y++) {
    for (let x = radius; x < width - radius; x++) {
      let sum = 0;
      for (let yy = -radius; yy <= radius; yy++) {
        for (let xx = -radius; xx <= radius; xx++) {
          sum += src[(y + yy) * width + (x + xx)];
        }
      }
      dst[y * width + x] = sum / k;
    }
  }
  return dst;
}

function harrisCorners(imageData, opts = {}) {
  const {
    k = 0.04,
    blurRadius = 2,
    responseThreshold = 1e6,
    maxCorners = 2500,
    nmsRadius = 6,
    scanStep = 2,
  } = opts;

  const { width, height } = imageData;
  const { gray } = imageDataToGrayscaleFloat(imageData);

  const { gx, gy } = convolveSobel(gray, width, height);

  const Ixx = new Float32Array(width * height);
  const Iyy = new Float32Array(width * height);
  const Ixy = new Float32Array(width * height);

  for (let i = 0; i < width * height; i++) {
    Ixx[i] = gx[i] * gx[i];
    Iyy[i] = gy[i] * gy[i];
    Ixy[i] = gx[i] * gy[i];
  }

  const Sxx = boxBlur(Ixx, width, height, blurRadius);
  const Syy = boxBlur(Iyy, width, height, blurRadius);
  const Sxy = boxBlur(Ixy, width, height, blurRadius);

  const R = new Float32Array(width * height);
  let Rmax = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const a = Sxx[idx];
      const b = Sxy[idx];
      const c = Syy[idx];

      const det = a * c - b * b;
      const trace = a + c;
      const r = det - k * trace * trace;

      R[idx] = r;
      if (r > Rmax) Rmax = r;
    }
  }

  const candidates = [];
  const thr = Math.max(responseThreshold, Rmax * 0.01);

  for (let y = 8; y < height - 8; y += scanStep) {
    for (let x = 8; x < width - 8; x += scanStep) {
      const idx = y * width + x;
      const r = R[idx];
      if (r > thr) candidates.push({ x, y, r });
    }
  }

  candidates.sort((a, b) => b.r - a.r);

  const selected = [];
  const taken = new Uint8Array(width * height);

  function markRadius(cx, cy, rad) {
    const r2 = rad * rad;
    for (let yy = -rad; yy <= rad; yy++) {
      for (let xx = -rad; xx <= rad; xx++) {
        if (xx * xx + yy * yy <= r2) {
          const nx = cx + xx;
          const ny = cy + yy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            taken[ny * width + nx] = 1;
          }
        }
      }
    }
  }

  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const idx = c.y * width + c.x;
    if (taken[idx]) continue;

    selected.push(c);
    markRadius(c.x, c.y, nmsRadius);

    if (selected.length >= maxCorners) break;
  }

  return { corners: selected, Rmax, thresholdUsed: thr };
}

async function drawDataUrlToCanvas(canvas, dataUrl) {
  const ctx = get2D(canvas, false);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve();
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

function drawCornersOverlay(canvas, baseDataUrl, corners, opts = {}) {
  const { radius = 3, color = "rgba(255,0,0,0.9)" } = opts;

  return new Promise((resolve, reject) => {
    const ctx = get2D(canvas, false);
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      ctx.fillStyle = color;
      for (const c of corners) {
        ctx.beginPath();
        ctx.arc(c.x, c.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = reject;
    img.src = baseDataUrl;
  });
}

function kmeansPixels(imageData, k = 5, maxIter = 10, sampleN = 15000) {
  const { data, width, height } = imageData;
  const total = width * height;

  const indices = [];
  const step = Math.max(1, Math.floor(total / sampleN));
  for (let i = 0; i < total; i += step) indices.push(i);

  const centers = [];
  for (let i = 0; i < k; i++) {
    const idx = indices[Math.floor(Math.random() * indices.length)];
    centers.push([data[idx * 4 + 0], data[idx * 4 + 1], data[idx * 4 + 2]]);
  }

  const labels = new Int32Array(indices.length);

  for (let iter = 0; iter < maxIter; iter++) {
    for (let i = 0; i < indices.length; i++) {
      const pi = indices[i];
      const pr = data[pi * 4 + 0];
      const pg = data[pi * 4 + 1];
      const pb = data[pi * 4 + 2];

      let best = 0;
      let bestD = Infinity;

      for (let c = 0; c < k; c++) {
        const cr = centers[c][0];
        const cg = centers[c][1];
        const cb = centers[c][2];
        const d = (pr - cr) ** 2 + (pg - cg) ** 2 + (pb - cb) ** 2;
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      labels[i] = best;
    }

    const sum = Array.from({ length: k }, () => [0, 0, 0, 0]);
    for (let i = 0; i < indices.length; i++) {
      const pi = indices[i];
      const lab = labels[i];
      sum[lab][0] += data[pi * 4 + 0];
      sum[lab][1] += data[pi * 4 + 1];
      sum[lab][2] += data[pi * 4 + 2];
      sum[lab][3] += 1;
    }

    for (let c = 0; c < k; c++) {
      if (sum[c][3] > 0) {
        centers[c][0] = sum[c][0] / sum[c][3];
        centers[c][1] = sum[c][1] / sum[c][3];
        centers[c][2] = sum[c][2] / sum[c][3];
      }
    }
  }

  const out = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < total; i++) {
    const pr = data[i * 4 + 0];
    const pg = data[i * 4 + 1];
    const pb = data[i * 4 + 2];

    let best = 0;
    let bestD = Infinity;

    for (let c = 0; c < k; c++) {
      const cr = centers[c][0];
      const cg = centers[c][1];
      const cb = centers[c][2];
      const d = (pr - cr) ** 2 + (pg - cg) ** 2 + (pb - cb) ** 2;
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }

    out[i * 4 + 0] = centers[best][0];
    out[i * 4 + 1] = centers[best][1];
    out[i * 4 + 2] = centers[best][2];
    out[i * 4 + 3] = 255;
  }

  return {
    clustered: new ImageData(out, width, height),
    centers: centers.map((c) => c.map((v) => Math.round(v))),
  };
}

// ---------- Backend calls ----------
async function fetchWithTimeout(url, options = {}, ms = 4500) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

async function uploadToBackend(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Backend upload failed (${res.status})`);
  return await res.json();
}

async function analyzeWithBackend(filename, options = {}) {
  const payload = {
    filename,
    options: {
      include_vit: false,
      include_llava_med: false,
      include_medgemma: false,
      include_images: false,
      ...options,
    },
  };

  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`Backend analyze failed: ${res.status} ${msg}`);
  }
  return await res.json();
}

// ---------- Report generator ----------
function buildReport({ file, client, backend }) {
  const lines = [];
  lines.push("3D MEDICAL MORPHOLOGY REVIEW REPORT");
  lines.push("==================================================");
  lines.push(`Generated: ${nowISO()}`);
  lines.push("");

  if (file) {
    lines.push("1. Study File");
    lines.push("--------------------------------------------------");
    lines.push(`Name: ${file.name}`);
    lines.push(`Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB`);
    lines.push(`Type: ${file.type || "application/octet-stream"}`);
    lines.push("");
  }

  if (client) {
    lines.push("2. Client-Side Mesh and Image Analysis");
    lines.push("--------------------------------------------------");
    lines.push(`Mesh format: ${client.stlFormat}`);
    lines.push(`Vertices: ${client.vertices.toLocaleString()}`);
    lines.push(`Faces: ${client.faces.toLocaleString()}`);
    lines.push(`Canvas size: ${client.canvasSize} x ${client.canvasSize}`);
    lines.push("");

    lines.push("2.1 Geometric / Visual Signal Summary");
    lines.push(`Edge density: ${(client.edgeDensity * 100).toFixed(2)}%`);
    lines.push(`Foreground ratio: ${(client.foregroundRatio * 100).toFixed(2)}%`);
    lines.push(`Harris corner count: ${client.harrisCount.toLocaleString()}`);
    lines.push(
      `Harris response max: ${formatNumber(client.harrisRmax, 2)} | threshold used: ${formatNumber(
        client.harrisThresholdUsed,
        2
      )}`
    );
    lines.push(`K-means clusters: ${client.kmeansK}`);
    lines.push(`Cluster centers: ${JSON.stringify(client.kmeansCenters)}`);
    lines.push("");
  }

  if (backend) {
    lines.push("3. Backend Morphology Analysis");
    lines.push("--------------------------------------------------");
    lines.push(`Status: ${backend.ok ? "OK" : "FAILED"}`);

    if (!backend.ok) {
      lines.push(`Error: ${backend.error || "unknown error"}`);
      lines.push("");
    } else {
      lines.push(`Device: ${backend.device || "unknown"}`);
      lines.push("");

      if (backend.geometry) {
        lines.push("3.1 Mesh Reconstruction Summary");
        lines.push(
          `Vertices: ${
            backend.geometry.vertices?.toLocaleString?.() ?? backend.geometry.vertices ?? "—"
          }`
        );
        lines.push(
          `Faces: ${backend.geometry.faces?.toLocaleString?.() ?? backend.geometry.faces ?? "—"}`
        );
        lines.push(`Volume: ${formatNumber(backend.geometry.volume, 2)}`);
        lines.push(`Surface area: ${formatNumber(backend.geometry.surface_area, 2)}`);
        lines.push(`Watertight: ${backend.geometry.is_watertight}`);
        if (backend.geometry.euler_number !== undefined) {
          lines.push(`Euler number: ${backend.geometry.euler_number}`);
        }
        lines.push("");
      }

      if (backend.patterns) {
        lines.push("3.2 Aggregate Morphology Signal Summary");
        lines.push(`Rendered views: ${backend.viewsCount}`);
        lines.push(`Total features (SIFT/ORB): ${backend.patterns.total_features}`);
        lines.push(
          `Average edge density: ${(backend.patterns.avg_edge_density * 100).toFixed(2)}%`
        );
        lines.push(`Total contours: ${backend.patterns.total_contours}`);
        lines.push("");
      }

      if (backend.medgemma) {
        lines.push("3.3 MedGemma Structured Morphology Review");
        lines.push(`Status: ${backend.medgemma.status || "ok"}`);
        if (backend.medgemma.model) {
          lines.push(`Model: ${backend.medgemma.model}`);
        }

        if (backend.medgemma.status === "ok") {
          const txt = coerceText(backend.medgemma.text).trim();
          if (txt) {
            lines.push("");
            lines.push(txt);
          } else {
            lines.push("No structured morphology text was generated.");
          }
        } else if (backend.medgemma.message) {
          lines.push(`Message: ${backend.medgemma.message}`);
        }

        lines.push("");
      }

      if (backend.llava_med) {
        lines.push("3.4 LLaVA-Med Supplemental Interpretation");
        lines.push(`Status: ${backend.llava_med.status || "ok"}`);

        if (backend.llava_med.status === "ok") {
          const txt = coerceText(backend.llava_med.text).trim();
          if (txt) {
            lines.push("");
            lines.push(txt);
          } else {
            lines.push("No supplemental interpretation text was generated.");
          }
        } else if (backend.llava_med.message) {
          lines.push(`Message: ${backend.llava_med.message}`);
        }

        lines.push("");
      }

      if (backend.vit) {
        lines.push("3.5 Optional Visual Classification (ViT)");
        lines.push(`Status: ${backend.vit.status || "ok"}`);
        if (backend.vit.predicted_class !== undefined) {
          lines.push(`Predicted class: ${backend.vit.predicted_class}`);
          lines.push(`Confidence: ${formatNumber(backend.vit.confidence, 4)}`);
        }
        if (backend.vit.label) lines.push(`Label: ${backend.vit.label}`);
        if (backend.vit.message) lines.push(`Message: ${backend.vit.message}`);
        lines.push("");
      }
    }
  }

  lines.push("4. Interpretive Caveats");
  lines.push("--------------------------------------------------");
  lines.push("• This report is based on reconstructed mesh geometry and rendered views, not raw CT, MRI, PET, ultrasound, or histopathology data.");
  lines.push("• The analysis focuses on external morphology, contour, asymmetry, branching, irregularity, and structural plausibility.");
  lines.push("• Internal tissue composition, radiodensity, enhancement behavior, and microscopic pathology cannot be established from surface mesh geometry alone.");
  lines.push("• AI-generated statements are intended for supportive morphology review and segmentation-quality interpretation, not diagnostic decision-making.");
  lines.push("");

  return lines.join("\n");
}

// ===================================================================
// ============================= COMPONENT ============================
// ===================================================================
export default function MedicalMeshCVAnalyzer() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);

  const [activeTab, setActiveTab] = useState("upload");
  const [processing, setProcessing] = useState(false);

  const [backendStatus, setBackendStatus] = useState("checking");
  const [useBackend, setUseBackend] = useState(true);
  const [backendHealth, setBackendHealth] = useState(null);

  const [results, setResults] = useState(null);

  const [layer, setLayer] = useState("edges");

  const [canvasSize, setCanvasSize] = useState(512);
  const [edgeThreshold, setEdgeThreshold] = useState(70);
  const [harrisMaxCorners, setHarrisMaxCorners] = useState(2500);
  const [kmeansK, setKmeansK] = useState(5);

  const [includeViT, setIncludeViT] = useState(true);
  const [includeLlavaMed, setIncludeLlavaMed] = useState(true);
  const [includeMedGemma, setIncludeMedGemma] = useState(true);
  const [includeBackendImages, setIncludeBackendImages] = useState(false);

  const canvasRef = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetchWithTimeout(`${API_URL}/health`, {}, 3500);
        if (!res.ok) {
          setBackendStatus("error");
          setUseBackend(false);
          return;
        }
        const data = await res.json();
        setBackendStatus("connected");
        setBackendHealth(data);
      } catch (e) {
        setBackendStatus("disconnected");
        setUseBackend(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (activeTab !== "results") return;
    if (!results?.images?.[layer]) return;

    const c = canvasRef.current;
    if (!c) return;

    const size = results?.client?.canvasSize ?? 512;
    if (c.width !== size) c.width = size;
    if (c.height !== size) c.height = size;

    drawDataUrlToCanvas(c, results.images[layer]).catch(() => {});
  }, [activeTab, layer, results]);

  const availableLayers = useMemo(() => {
    if (!results?.images) return [];
    const order = ["original", "sobel", "edges", "corners", "mask", "kmeans"];
    return order.filter((k) => results.images[k]);
  }, [results]);

  const handleFileUpload = (e) => {
    const uploaded = Array.from(e.target.files || []);
    const meshes = uploaded.filter((f) => {
    const name = f.name.toLowerCase();
    return name.endsWith(".stl") || name.endsWith(".obj");
  });

  setFiles(meshes);
  if (meshes.length > 0) setSelectedFile(meshes[0]);
};
  const resetAll = () => {
    setResults(null);
    setActiveTab("upload");
    setLayer("edges");
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    setProcessing(true);
    setActiveTab("results");
    await new Promise((r) => setTimeout(r, 20));

    const size = clamp(Number(canvasSize) || 512, 512, 2048);

    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;

    const ctx = get2D(canvas, true);

    let backendRaw = null;

    try {
      if (useBackend && backendStatus === "connected") {
        try {
          await uploadToBackend(selectedFile);
          backendRaw = await analyzeWithBackend(selectedFile.name, {
            include_vit: includeViT,
            include_llava_med: includeLlavaMed,
            include_medgemma: includeMedGemma,
            include_images: includeBackendImages,
          });
        } catch (err) {
          backendRaw = { ok: false, error: err?.message || String(err) };
        }
      }

      const lowerName = selectedFile.name.toLowerCase();

      let mesh;
      if (lowerName.endsWith(".stl")) {
        mesh = await parseSTL(selectedFile);
      } else if (lowerName.endsWith(".obj")) {
        mesh = await parseOBJ(selectedFile);
      } else {
        throw new Error("Unsupported mesh format. Please upload STL or OBJ.");
      }

      const { vertices, faces, format } = mesh;

      const originalDataUrl = renderMeshToCanvas(vertices, faces, canvas, {
        maxFacesToDraw: 120_000,
        strokeColor: "#00ff00",
      });

      const baseImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      const sobelImageData = sobelMagnitudeImageData(baseImageData);
      ctx.putImageData(sobelImageData, 0, 0);
      const sobelDataUrl = canvas.toDataURL("image/png");

      const { edgeImageData, edgeDensity } = edgeMapFromSobel(baseImageData, edgeThreshold);
      ctx.putImageData(edgeImageData, 0, 0);
      const edgesDataUrl = canvas.toDataURL("image/png");

      const { mask, foregroundRatio } = foregroundSegmentation(baseImageData, 15);
      ctx.putImageData(mask, 0, 0);
      const maskDataUrl = canvas.toDataURL("image/png");

      const harris = harrisCorners(baseImageData, { maxCorners: harrisMaxCorners });
      const cornersOverlayUrl = await drawCornersOverlay(canvas, originalDataUrl, harris.corners, {
        radius: 2,
      });

      const km = kmeansPixels(baseImageData, kmeansK, 10, 15000);
      ctx.putImageData(km.clustered, 0, 0);
      const kmeansUrl = canvas.toDataURL("image/png");

      const client = {
        stlFormat: format,
        vertices: vertices.length,
        faces: faces.length,
        canvasSize: size,
        edgeThreshold,
        edgeDensity,
        foregroundRatio,
        harrisCount: harris.corners.length,
        harrisRmax: harris.Rmax,
        harrisThresholdUsed: harris.thresholdUsed,
        kmeansK,
        kmeansCenters: km.centers,
      };

      const normalizedBackend = backendRaw
        ? {
            ok: backendRaw.ok === false ? false : !backendRaw.error,
            error: backendRaw.error,
            device: backendRaw.device || backendHealth?.device || "unknown",
            geometry: backendRaw.geometry,
            patterns: backendRaw.patterns,
            viewsCount: backendRaw.views?.length || 0,
            views: backendRaw.views || [],
            vit: backendRaw.vit || backendRaw.vision_transformer || backendRaw.vit_result || null,
            llava_med: backendRaw.llava_med || backendRaw.llava_med_result || backendRaw.vlm || null,
            medgemma: backendRaw.medgemma || null,
          }
        : { ok: false, error: "Backend disabled" };

      if (normalizedBackend.llava_med) {
        normalizedBackend.llava_med = {
          ...normalizedBackend.llava_med,
          text: coerceText(normalizedBackend.llava_med.text),
          message: normalizedBackend.llava_med.message
            ? String(normalizedBackend.llava_med.message)
            : undefined,
          status: normalizedBackend.llava_med.status
            ? String(normalizedBackend.llava_med.status)
            : "ok",
        };
      }

      if (normalizedBackend.medgemma) {
        normalizedBackend.medgemma = {
          ...normalizedBackend.medgemma,
          text: coerceText(normalizedBackend.medgemma.text),
          message: normalizedBackend.medgemma.message
            ? String(normalizedBackend.medgemma.message)
            : undefined,
          status: normalizedBackend.medgemma.status
            ? String(normalizedBackend.medgemma.status)
            : "ok",
          model: normalizedBackend.medgemma.model
            ? String(normalizedBackend.medgemma.model)
            : undefined,
        };
      }

      const report = buildReport({
        file: selectedFile,
        client,
        backend: normalizedBackend,
      });

      const nextResults = {
        file: { name: selectedFile.name, size: selectedFile.size, stlFormat: format },
        images: {
          original: originalDataUrl,
          sobel: sobelDataUrl,
          edges: edgesDataUrl,
          corners: cornersOverlayUrl,
          mask: maskDataUrl,
          kmeans: kmeansUrl,
        },
        client,
        backend: normalizedBackend,
        report,
        generated_at: nowISO(),
      };

      setResults(nextResults);
      setLayer("edges");

      const display = canvasRef.current;
      if (display) {
        display.width = size;
        display.height = size;
        await drawDataUrlToCanvas(display, nextResults.images.edges);
      }
    } catch (error) {
      console.error("❌ Analysis error:", error);
      alert("Error processing file: " + (error?.message || String(error)));
    } finally {
      setProcessing(false);
    }
  };

  const downloadAll = () => {
    if (!results) return;
    downloadJSON(`analysis-${results.file.name}.json`, results);
    downloadText(`report-${results.file.name}.txt`, results.report);
  };

  const downloadReportOnly = () => {
    if (!results) return;
    downloadText(`report-${results.file.name}.txt`, results.report);
  };

  const downloadJSONOnly = () => {
    if (!results) return;
    downloadJSON(`analysis-${results.file.name}.json`, results);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-gray-800 rounded-lg shadow-2xl overflow-hidden">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6">
            <div className="flex justify-between items-start gap-6">
              <div>
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                  <Layers className="w-8 h-8" />
                  Albabish Medical Mesh CV Analyzer
                </h1>
                <p className="text-blue-100 mt-2">
                  Mesh → 2D Projection → Computer Vision + optional backend morphology models
                  (ViT / LLaVA-Med / MedGemma)
                </p>
              </div>

              <div className="flex flex-col gap-2 items-end">
                <div
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                    backendStatus === "connected"
                      ? "bg-green-500 bg-opacity-20"
                      : "bg-yellow-500 bg-opacity-20"
                  }`}
                >
                  {backendStatus === "connected" ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-300" />
                      <span className="text-white text-sm font-medium">Backend Connected</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-5 h-5 text-yellow-300" />
                      <span className="text-white text-sm font-medium">Client-Side Mode</span>
                    </>
                  )}
                </div>

                {backendStatus === "connected" && (
                  <button
                    onClick={() => setUseBackend((s) => !s)}
                    className={`text-xs px-3 py-1 rounded ${
                      useBackend ? "bg-blue-500 text-white" : "bg-gray-600 text-gray-300"
                    }`}
                  >
                    <Server className="w-3 h-3 inline mr-1" />
                    {useBackend ? "Using Backend" : "Client Only"}
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="p-6">
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => setActiveTab("upload")}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition ${
                  activeTab === "upload"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
              >
                <Upload className="w-4 h-4" />
                Upload
              </button>

              <button
                onClick={() => setActiveTab("results")}
                className={`px-4 py-2 rounded-lg flex items-center gap-2 transition ${
                  activeTab === "results"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
                disabled={!results}
              >
                <Eye className="w-4 h-4" />
                Results
              </button>

              {results && (
                <button
                  onClick={resetAll}
                  className="ml-auto px-4 py-2 rounded-lg bg-gray-700 text-gray-200 hover:bg-gray-600 transition"
                >
                  Reset
                </button>
              )}
            </div>

            {activeTab === "upload" && (
              <div className="space-y-6">
                <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-blue-500 transition">
                  <input
                    type="file"
                    multiple
                    accept=".stl,.obj"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer flex flex-col items-center gap-4"
                  >
                    <Upload className="w-16 h-16 text-blue-400" />
                    <div>
                      <p className="text-white text-lg font-semibold">Upload STL or OBJ Files</p>
                      <p className="text-gray-400 text-sm mt-1">Click to browse or drag and drop</p>
                    </div>
                  </label>
                </div>

                {files.length > 0 && (
                  <div className="bg-gray-700 rounded-lg p-4">
                    <h3 className="text-white font-semibold mb-3">Uploaded Files ({files.length})</h3>
                    <div className="space-y-2">
                      {files.map((file, idx) => (
                        <div
                          key={idx}
                          onClick={() => setSelectedFile(file)}
                          className={`p-3 rounded cursor-pointer transition ${
                            selectedFile === file ? "bg-blue-600" : "bg-gray-600 hover:bg-gray-500"
                          }`}
                        >
                          <p className="text-white font-medium">{file.name}</p>
                          <p className="text-gray-300 text-sm">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3">Analysis Settings</h3>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="bg-gray-600 p-3 rounded">
                      <label className="text-gray-300 text-sm">Canvas Size</label>
                      <input
                        type="number"
                        value={canvasSize}
                        onChange={(e) => setCanvasSize(Number(e.target.value))}
                        className="mt-1 w-full rounded bg-gray-800 text-white px-2 py-1"
                        min={512}
                        max={2048}
                      />
                      <p className="text-gray-300 text-xs mt-1">Higher = more detail, slower.</p>
                    </div>

                    <div className="bg-gray-600 p-3 rounded">
                      <label className="text-gray-300 text-sm">Edge Threshold</label>
                      <input
                        type="number"
                        value={edgeThreshold}
                        onChange={(e) => setEdgeThreshold(Number(e.target.value))}
                        className="mt-1 w-full rounded bg-gray-800 text-white px-2 py-1"
                        min={0}
                        max={255}
                      />
                      <p className="text-gray-300 text-xs mt-1">Sobel magnitude threshold.</p>
                    </div>

                    <div className="bg-gray-600 p-3 rounded">
                      <label className="text-gray-300 text-sm">Max Harris Corners</label>
                      <input
                        type="number"
                        value={harrisMaxCorners}
                        onChange={(e) => setHarrisMaxCorners(Number(e.target.value))}
                        className="mt-1 w-full rounded bg-gray-800 text-white px-2 py-1"
                        min={100}
                        max={20000}
                      />
                      <p className="text-gray-300 text-xs mt-1">Higher = more points, slower.</p>
                    </div>

                    <div className="bg-gray-600 p-3 rounded">
                      <label className="text-gray-300 text-sm">K-means Clusters</label>
                      <input
                        type="number"
                        value={kmeansK}
                        onChange={(e) => setKmeansK(Number(e.target.value))}
                        className="mt-1 w-full rounded bg-gray-800 text-white px-2 py-1"
                        min={2}
                        max={10}
                      />
                      <p className="text-gray-300 text-xs mt-1">Pixel clustering visualization.</p>
                    </div>

                    <div className="bg-gray-600 p-3 rounded md:col-span-2">
                      <label className="text-gray-300 text-sm">
                        Backend Models (ViT / LLaVA-Med / MedGemma)
                      </label>

                      <div className="mt-2 flex flex-wrap items-center gap-3">
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            backendStatus === "connected"
                              ? "bg-green-600 text-white"
                              : "bg-gray-800 text-gray-300"
                          }`}
                        >
                          {backendStatus === "connected" ? "Available" : "Unavailable"}
                        </span>

                        <label className="flex items-center gap-2 text-xs text-gray-200">
                          <input
                            type="checkbox"
                            checked={includeViT}
                            onChange={(e) => setIncludeViT(e.target.checked)}
                            className="accent-blue-500"
                          />
                          ViT
                        </label>

                        <label className="flex items-center gap-2 text-xs text-gray-200">
                          <input
                            type="checkbox"
                            checked={includeLlavaMed}
                            onChange={(e) => setIncludeLlavaMed(e.target.checked)}
                            className="accent-purple-500"
                          />
                          LLaVA-Med
                        </label>

                        <label className="flex items-center gap-2 text-xs text-gray-200">
                          <input
                            type="checkbox"
                            checked={includeMedGemma}
                            onChange={(e) => setIncludeMedGemma(e.target.checked)}
                            className="accent-pink-500"
                          />
                          MedGemma
                        </label>

                        <label className="flex items-center gap-2 text-xs text-gray-200">
                          <input
                            type="checkbox"
                            checked={includeBackendImages}
                            onChange={(e) => setIncludeBackendImages(e.target.checked)}
                            className="accent-green-500"
                          />
                          Backend base64 images (large)
                        </label>

                        <span className="text-gray-300 text-xs">
                          Text outputs are shown only when the backend returns status <code>ok</code>.
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {selectedFile && (
                  <button
                    onClick={runAnalysis}
                    disabled={processing}
                    className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg font-semibold flex items-center justify-center gap-2 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transition"
                  >
                    {processing ? (
                      <>Processing...</>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        Run Full CV Analysis
                      </>
                    )}
                  </button>
                )}
              </div>
            )}

            {activeTab === "results" && (
              <div className="space-y-6">
                <div className="flex flex-wrap justify-between items-center gap-3">
                  <h2 className="text-2xl font-bold text-white">Analysis Results</h2>

                  <div className="flex gap-2">
                    <button
                      onClick={downloadJSONOnly}
                      disabled={!results}
                      className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition disabled:opacity-50"
                    >
                      <Download className="w-4 h-4" />
                      Download JSON
                    </button>

                    <button
                      onClick={downloadReportOnly}
                      disabled={!results}
                      className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition disabled:opacity-50"
                    >
                      <FileText className="w-4 h-4" />
                      Report TXT
                    </button>

                    <button
                      onClick={downloadAll}
                      disabled={!results}
                      className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition disabled:opacity-50"
                    >
                      <Download className="w-4 h-4" />
                      All
                    </button>
                  </div>
                </div>

                {processing && (
                  <div className="text-center py-6 bg-gray-700 rounded-lg">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                    <p className="text-white mt-4">Processing your file...</p>
                    <p className="text-gray-300 text-sm mt-2">
                      Executing mesh morphology pipeline: render → edges → Harris corners →
                      segmentation → optional backend (ViT / LLaVA-Med / MedGemma)...
                    </p>
                  </div>
                )}

                {results && !processing && (
                  <div className="grid lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 space-y-4">
                      <div className="bg-gray-700 rounded-lg p-4">
                        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                          <div className="flex items-center gap-2">
                            <ImageIcon className="w-5 h-5 text-blue-300" />
                            <h3 className="text-white font-semibold">Visualization</h3>
                          </div>

                          <div className="flex flex-wrap gap-2">
                            {availableLayers.map((k) => (
                              <button
                                key={k}
                                onClick={() => setLayer(k)}
                                className={`text-xs px-3 py-1 rounded ${
                                  layer === k
                                    ? "bg-blue-600 text-white"
                                    : "bg-gray-600 text-gray-200 hover:bg-gray-500"
                                }`}
                              >
                                {k.toUpperCase()}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="border border-gray-600 rounded overflow-hidden">
                          <canvas ref={canvasRef} className="w-full" />
                        </div>

                        <p className="text-gray-300 text-xs mt-2">
                          Layer: <span className="text-white font-semibold">{layer}</span>
                        </p>
                      </div>

                      <div className="bg-gray-700 rounded-lg p-4">
                        <h3 className="text-white font-semibold mb-3">Backend Multi-View Summary</h3>

                        {!results.backend?.ok ? (
                          <p className="text-gray-300 text-sm">
                            Backend disabled or failed. Using client-only pipeline.
                            {results.backend?.error ? (
                              <span className="block mt-2 text-xs text-red-200">
                                Error: {results.backend.error}
                              </span>
                            ) : null}
                          </p>
                        ) : (
                          <>
                            <div className="grid md:grid-cols-2 gap-4">
                              <div className="bg-gray-600 p-3 rounded">
                                <p className="text-gray-300 text-sm">Views</p>
                                <p className="text-white text-lg font-bold">{results.backend.viewsCount}</p>
                              </div>

                              <div className="bg-gray-600 p-3 rounded">
                                <p className="text-gray-300 text-sm">Total Features (SIFT/ORB)</p>
                                <p className="text-white text-lg font-bold">
                                  {results.backend.patterns?.total_features ?? "—"}
                                </p>
                              </div>

                              <div className="bg-gray-600 p-3 rounded">
                                <p className="text-gray-300 text-sm">Avg Edge Density</p>
                                <p className="text-white text-lg font-bold">
                                  {results.backend.patterns?.avg_edge_density !== undefined
                                    ? `${(results.backend.patterns.avg_edge_density * 100).toFixed(2)}%`
                                    : "—"}
                                </p>
                              </div>

                              <div className="bg-gray-600 p-3 rounded">
                                <p className="text-gray-300 text-sm">Total Contours</p>
                                <p className="text-white text-lg font-bold">
                                  {results.backend.patterns?.total_contours ?? "—"}
                                </p>
                              </div>
                            </div>

                            {results.backend.views?.length > 0 && (
                              <div className="mt-4 overflow-x-auto">
                                <table className="w-full text-sm text-gray-200">
                                  <thead>
                                    <tr className="text-gray-300">
                                      <th className="text-left py-2 pr-2">View</th>
                                      <th className="text-left py-2 pr-2">Keypoints</th>
                                      <th className="text-left py-2 pr-2">Contours</th>
                                      <th className="text-left py-2 pr-2">Clusters</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {results.backend.views.map((v) => (
                                      <tr key={v.view_index} className="border-t border-gray-600">
                                        <td className="py-2 pr-2">{v.view_index}</td>
                                        <td className="py-2 pr-2">{v.features?.keypoints_detected ?? "—"}</td>
                                        <td className="py-2 pr-2">{v.contours?.num_contours ?? "—"}</td>
                                        <td className="py-2 pr-2">{v.segmentation?.clusters ?? "—"}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-gray-700 rounded-lg p-4">
                        <h3 className="text-white font-semibold mb-3">Analysis Statistics</h3>

                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-gray-600 p-3 rounded">
                            <p className="text-gray-300 text-xs">Vertices</p>
                            <p className="text-white text-lg font-bold">
                              {results.client.vertices.toLocaleString()}
                            </p>
                          </div>

                          <div className="bg-gray-600 p-3 rounded">
                            <p className="text-gray-300 text-xs">Faces</p>
                            <p className="text-white text-lg font-bold">
                              {results.client.faces.toLocaleString()}
                            </p>
                          </div>

                          <div className="bg-gray-600 p-3 rounded">
                            <p className="text-gray-300 text-xs">Edge Density</p>
                            <p className="text-white text-lg font-bold">
                              {(results.client.edgeDensity * 100).toFixed(2)}%
                            </p>
                          </div>

                          <div className="bg-gray-600 p-3 rounded">
                            <p className="text-gray-300 text-xs">Harris Corners</p>
                            <p className="text-white text-lg font-bold">
                              {results.client.harrisCount.toLocaleString()}
                            </p>
                          </div>

                          <div className="bg-gray-600 p-3 rounded col-span-2">
                            <p className="text-gray-300 text-xs">Foreground Ratio</p>
                            <p className="text-white text-lg font-bold">
                              {(results.client.foregroundRatio * 100).toFixed(2)}%
                            </p>
                          </div>
                        </div>

                        {results.backend?.ok && results.backend.geometry && (
                          <div className="mt-4 grid grid-cols-2 gap-3">
                            <div className="bg-gray-600 p-3 rounded">
                              <p className="text-gray-300 text-xs">Volume</p>
                              <p className="text-white font-bold">
                                {formatNumber(results.backend.geometry.volume, 2)}
                              </p>
                            </div>

                            <div className="bg-gray-600 p-3 rounded">
                              <p className="text-gray-300 text-xs">Surface Area</p>
                              <p className="text-white font-bold">
                                {formatNumber(results.backend.geometry.surface_area, 2)}
                              </p>
                            </div>

                            <div className="bg-gray-600 p-3 rounded col-span-2">
                              <p className="text-gray-300 text-xs">Watertight</p>
                              <p className="text-white font-bold">
                                {String(results.backend.geometry.is_watertight)}
                              </p>
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="bg-gray-700 rounded-lg p-4">
                        <h3 className="text-white font-semibold mb-2">Vision Transformer (ViT)</h3>

                        {!results.backend?.ok ? (
                          <p className="text-gray-300 text-sm">Backend not available, ViT not executed.</p>
                        ) : results.backend?.vit ? (
                          <div className="bg-gray-600 rounded p-3 text-gray-200 text-sm">
                            <p>
                              <span className="text-gray-300">Status:</span>{" "}
                              <span className="text-white font-semibold">
                                {results.backend.vit.status || "ok"}
                              </span>
                            </p>

                            {results.backend.vit.predicted_class !== undefined && (
                              <p>
                                <span className="text-gray-300">Predicted class:</span>{" "}
                                <span className="text-white font-semibold">
                                  {results.backend.vit.predicted_class}
                                </span>
                              </p>
                            )}

                            {results.backend.vit.confidence !== undefined && (
                              <p>
                                <span className="text-gray-300">Confidence:</span>{" "}
                                <span className="text-white font-semibold">
                                  {formatNumber(results.backend.vit.confidence, 4)}
                                </span>
                              </p>
                            )}

                            {results.backend.vit.label && (
                              <p>
                                <span className="text-gray-300">Label:</span>{" "}
                                <span className="text-white font-semibold">
                                  {results.backend.vit.label}
                                </span>
                              </p>
                            )}

                            {results.backend.vit.message && (
                              <p className="text-gray-300 mt-2">{results.backend.vit.message}</p>
                            )}
                          </div>
                        ) : (
                          <p className="text-gray-300 text-sm">
                            Backend did not return ViT output. Ensure Flask returns <code>vit</code>.
                          </p>
                        )}
                      </div>

                      <div className="bg-gray-700 rounded-lg p-4">
                        <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
                          <Brain className="w-5 h-5 text-purple-300" />
                          LLaVA-Med (Supplemental)
                        </h3>

                        {!results.backend?.ok ? (
                          <p className="text-gray-300 text-sm">
                            Backend not available, LLaVA-Med not executed.
                          </p>
                        ) : results.backend?.llava_med ? (
                          <div className="bg-gray-600 rounded p-3 text-gray-200 text-sm">
                            <p>
                              <span className="text-gray-300">Status:</span>{" "}
                              <span className="text-white font-semibold">
                                {results.backend.llava_med.status || "ok"}
                              </span>
                            </p>

                            {results.backend.llava_med.status !== "ok" ? (
                              results.backend.llava_med.message ? (
                                <p className="text-gray-300 mt-2">{results.backend.llava_med.message}</p>
                              ) : (
                                <p className="text-gray-300 mt-2 text-xs">No error message returned.</p>
                              )
                            ) : results.backend.llava_med.text?.trim() ? (
                              <pre className="mt-3 text-xs text-gray-200 bg-gray-800 p-3 rounded overflow-auto max-h-[320px] whitespace-pre-wrap">
                                {results.backend.llava_med.text}
                              </pre>
                            ) : (
                              <p className="text-gray-300 mt-2 text-xs">No text was generated.</p>
                            )}
                          </div>
                        ) : (
                          <p className="text-gray-300 text-sm">
                            Backend did not return LLaVA-Med output. Ensure Flask returns <code>llava_med</code>.
                          </p>
                        )}
                      </div>

                      <div className="bg-gray-700 rounded-lg p-4">
                        <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
                          <Brain className="w-5 h-5 text-pink-300" />
                          MedGemma (Morphology Review)
                        </h3>

                        {!results.backend?.ok ? (
                          <p className="text-gray-300 text-sm">Backend not available, MedGemma not executed.</p>
                        ) : results.backend?.medgemma ? (
                          <div className="bg-gray-600 rounded p-3 text-gray-200 text-sm">
                            <p>
                              <span className="text-gray-300">Status:</span>{" "}
                              <span className="text-white font-semibold">
                                {results.backend.medgemma.status || "ok"}
                              </span>
                            </p>

                            {results.backend.medgemma.model && (
                              <p>
                                <span className="text-gray-300">Model:</span>{" "}
                                <span className="text-white font-semibold">
                                  {results.backend.medgemma.model}
                                </span>
                              </p>
                            )}

                            {results.backend.medgemma.status !== "ok" ? (
                              results.backend.medgemma.message ? (
                                <p className="text-gray-300 mt-2">{results.backend.medgemma.message}</p>
                              ) : (
                                <p className="text-gray-300 mt-2 text-xs">No error message returned.</p>
                              )
                            ) : results.backend.medgemma.text?.trim() ? (
                              <pre className="mt-3 text-xs text-gray-200 bg-gray-800 p-3 rounded overflow-auto max-h-[320px] whitespace-pre-wrap">
                                {results.backend.medgemma.text}
                              </pre>
                            ) : (
                              <p className="text-gray-300 mt-2 text-xs">No text was generated.</p>
                            )}
                          </div>
                        ) : (
                          <p className="text-gray-300 text-sm">
                            Backend did not return MedGemma output. Ensure Flask returns{" "}
                            <code>medgemma</code>.
                          </p>
                        )}
                      </div>

                      <div className="bg-gray-700 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-white font-semibold">Morphology Report</h3>
                          <button
                            onClick={() => navigator.clipboard.writeText(results.report).catch(() => {})}
                            className="text-xs px-3 py-1 rounded bg-gray-600 text-gray-200 hover:bg-gray-500"
                          >
                            Copy
                          </button>
                        </div>

                        <pre className="text-xs text-gray-200 bg-gray-800 p-3 rounded overflow-auto max-h-[360px] whitespace-pre-wrap">
                          {results.report}
                        </pre>
                      </div>
                    </div>
                  </div>
                )}

                {!results && !processing && (
                  <div className="text-center py-12 bg-gray-700 rounded-lg">
                    <p className="text-gray-300">No results yet. Upload a file and run analysis.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="mt-6 bg-gray-800 rounded-lg p-6">
          <h3 className="text-white font-semibold mb-3">Algorithms Applied</h3>

          <div className="grid md:grid-cols-2 gap-4 text-gray-300 text-sm">
            <div className="bg-gray-700 p-4 rounded">
              <h4 className="text-blue-300 font-semibold mb-2">🔍 Edge Detection</h4>
              <p>True Sobel gradient magnitude + threshold edge-map + edge density statistics</p>
            </div>

            <div className="bg-gray-700 p-4 rounded">
              <h4 className="text-blue-300 font-semibold mb-2">📍 Corner/Feature Detection</h4>
              <p>True Harris response + non-maximum suppression (NMS) + overlay visualization</p>
            </div>

            <div className="bg-gray-700 p-4 rounded">
              <h4 className="text-blue-300 font-semibold mb-2">🎨 Segmentation</h4>
              <p>Foreground mask from threshold + k-means pixel clustering visualization</p>
            </div>

            <div className="bg-gray-700 p-4 rounded">
              <h4 className="text-purple-300 font-semibold mb-2">🧠 Backend Morphology Models</h4>
              <p>Multi-view CV stats + ViT + LLaVA-Med + MedGemma for supportive 3D morphology review</p>
            </div>
          </div>

          <p className="text-gray-400 text-xs mt-4">
            Note: Client-side CV operates on a wireframe 2D projection. Backend analysis operates on rendered mesh views and provides supportive morphology interpretation, not diagnosis.
          </p>
        </div>
      </div>
    </div>
  );
}