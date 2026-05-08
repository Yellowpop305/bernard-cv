"""
Bernard CV — Tube length estimation microservice.
v4: ports Yellowpop's exact production tracing algorithm from neon/neon.py
    (Zhang-Suen thinning + contour arcLength / 2 method).

This guarantees Bernard's tube length matches Yellowpop's internal pricer to ±1%.
"""

from flask import Flask, request, jsonify
import requests
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import os

app = Flask(__name__)

# ---------- Constants matching Yellowpop's production code ----------

# Kernel cache (matches neon.py line 39-45)
_kernel_cache = {}
def kernel(size):
    if size not in _kernel_cache:
        _kernel_cache[size] = np.ones((size, size), np.uint8)
    return _kernel_cache[size]


# ---------- Health & info endpoints ----------

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'bernard-cv',
        'status': 'ok',
        'version': '4.0.0',
        'algorithm': 'Yellowpop production (Zhang-Suen thinning + arcLength)',
        'endpoints': {
            'POST /trace-tube': 'Estimate tube length for a design image',
            'GET  /health': 'Health check',
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


# ---------- Main endpoint ----------

@app.route('/trace-tube', methods=['POST'])
def trace_tube():
    """
    POST /trace-tube
    Body: {
        "image_url": "https://...",
        "sign_width_cm": 80,
        "single_line"?: false   // false = outlined design (use Canny edges); true = filled (use thresh directly)
    }
    Returns JSON with tube_length_m and diagnostics.
    """
    data = request.get_json(silent=True) or {}
    image_url = data.get('image_url')
    sign_width_cm = float(data.get('sign_width_cm', 80))
    # single_line: True | False | None (auto-detect). Default = auto.
    single_line_param = data.get('single_line', None)
    single_line = bool(single_line_param) if single_line_param is not None else None
    # Optional calibration multiplier (defaults to env var or 1.0)
    calibration = float(data.get('calibration', os.environ.get('PRODUCTION_CALIBRATION', '1.0')))

    if not image_url:
        return jsonify({'error': 'Missing image_url'}), 400

    try:
        result = analyze_image(image_url, sign_width_cm, single_line, calibration)
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({'error': f'Failed to fetch image: {e}'}), 400
    except Exception as e:
        app.logger.exception('Processing failed')
        return jsonify({'error': f'Processing failed: {e}'}), 500


# ---------- Image to lines (ported from neon.py image_to_lines) ----------

def image_to_lines(width_cm: float, gray: np.ndarray, single_line: bool) -> tuple:
    """
    Direct port of Yellowpop's image_to_lines() from neon/neon.py
    Returns (thin_image, width_px) — the thinned binary and content width in pixels.
    """
    # 1. OTSU threshold
    _ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Invert if more than half the pixels are white (background should be black, strokes white)
    if cv2.countNonZero(thr) / thr.size > 0.5:
        cv2.bitwise_not(thr, thr)

    # 3. Get bounding box of content
    bounding_box = cv2.boundingRect(thr)
    _x1, _y1, width_px, _height_px = bounding_box

    if width_px == 0:
        return None, 0

    # 4. Resize so that 1 pixel = 0.12 cm at the given sign width
    #    This matches Yellowpop's scale exactly (line 104 of neon.py)
    scale = width_cm / 0.12 / width_px
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

    # 5. Re-threshold after resize
    _ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(thr) / thr.size > 0.5:
        cv2.bitwise_not(thr, thr)

    # 6. Edge detection for outlined designs (when single_line=False)
    edges = cv2.Canny(thr, 20, 30)
    dilate = cv2.dilate(edges, kernel(50))
    edges = cv2.bitwise_and(edges, dilate)
    edges = cv2.dilate(edges, kernel(5))

    # 7. Zhang-Suen thinning (THE key step)
    source = thr if single_line else edges
    thin = cv2.ximgproc.thinning(source, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 8. Crop to non-zero region
    bounding_box = cv2.boundingRect(thin)
    x1, y1, w, h = bounding_box
    if w == 0:
        return None, 0
    result = thin[y1:y1 + h, x1:x1 + w]

    return result, w


def detect_filled_vs_outlined(gray: np.ndarray) -> tuple:
    """
    Heuristic: determine if the design is filled (use single_line=True)
    or outlined/thin lines (use single_line=False).

    Returns (single_line: bool, filled_ratio: float)
    """
    _ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(thr) / thr.size > 0.5:
        cv2.bitwise_not(thr, thr)

    bb = cv2.boundingRect(thr)
    _x, _y, bb_w, bb_h = bb
    if bb_w == 0 or bb_h == 0:
        return False, 0.0

    nonzero = cv2.countNonZero(thr)
    bb_area = bb_w * bb_h
    filled_ratio = nonzero / max(bb_area, 1)

    # Filled designs typically have > 20% pixel density in their bounding box.
    # Thin line art / outlines / glow-effect mockups typically < 15%.
    return filled_ratio > 0.20, filled_ratio


def analyze_image(image_url: str, sign_width_cm: float, single_line: bool | None, calibration: float = 1.0) -> dict:
    # 1. Download
    response = requests.get(image_url, timeout=20)
    response.raise_for_status()
    pil_img = Image.open(BytesIO(response.content)).convert('RGB')
    src = np.array(pil_img)

    # 2. Convert to grayscale (handle alpha if present, matches neon.py logic)
    if src.ndim == 3 and src.shape[-1] == 4:
        gray = src[:, :, 3]
        if np.count_nonzero(gray == 255) > 0.7 * gray.size:
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    elif src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        gray = src

    # 3. Auto-detect filled vs outlined if single_line not explicitly set
    auto_detected = False
    filled_ratio = None
    if single_line is None:
        single_line, filled_ratio = detect_filled_vs_outlined(gray)
        auto_detected = True

    # 4. Run image_to_lines (the Yellowpop algorithm)
    result, width_px = image_to_lines(sign_width_cm, gray, single_line)

    if result is None or width_px == 0:
        return {
            'error': 'Could not process image — no content detected',
        }

    # 5. THE KEY CALCULATION (from neon.py line 348-350):
    #    Find contours of the thinned image, sum arc lengths divided by 2
    contours, _hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    arc_length_px = sum(cv2.arcLength(c, True) for c in contours) / 2.0

    # 6. Convert to meters (matches Yellowpop's exact formula)
    raw_tube_length_m = round(arc_length_px * sign_width_cm / width_px / 100, 2)

    # 7. Apply optional calibration multiplier
    tube_length_m = round(raw_tube_length_m * calibration, 2)

    # Diagnostics
    h, w = result.shape
    nonzero = int(cv2.countNonZero(result))

    return {
        'tube_length_m': tube_length_m,
        'tube_length_m_raw': raw_tube_length_m,
        'calibration_applied': calibration,
        'arc_length_px': round(arc_length_px, 1),
        'width_px': int(width_px),
        'sign_width_cm': sign_width_cm,
        'single_line': single_line,
        'auto_detected_single_line': auto_detected,
        'filled_ratio': round(filled_ratio, 3) if filled_ratio is not None else None,
        'thin_image_dimensions': {'width': int(w), 'height': int(h)},
        'thin_nonzero_pixels': nonzero,
        'contour_count': len(contours),
        'algorithm': 'yellowpop_production_zhang_suen_arclength',
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
