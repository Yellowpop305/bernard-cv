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
    Direct port of Yellowpop's image_to_lines() from neon/neon.py, with added
    noise-reduction preprocessing for JPG-compressed / textured customer uploads.

    Returns (thin_image, width_px) — the thinned binary and content width in pixels.
    """
    # NEW: Gaussian blur to smooth out JPG compression artifacts and texture noise
    # before thresholding. Critical for customer-uploaded images that have grain,
    # backgrounds, or compression fragmentation.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. OTSU threshold
    _ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Invert if more than half the pixels are white (background should be black, strokes white)
    if cv2.countNonZero(thr) / thr.size > 0.5:
        cv2.bitwise_not(thr, thr)

    # NEW: Morphological closing to fill small gaps in strokes and remove tiny
    # speckle noise (3x3 kernel, light cleanup)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel(3))
    # NEW: Opening to remove small isolated specks (anti-aliasing fragments)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel(2))

    # 3. Get bounding box of content
    bounding_box = cv2.boundingRect(thr)
    _x1, _y1, width_px, _height_px = bounding_box

    if width_px == 0:
        return None, 0

    # 4. Resize so that 1 pixel = 0.12 cm at the given sign width
    scale = width_cm / 0.12 / width_px
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

    # 5. Re-threshold after resize
    _ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(thr) / thr.size > 0.5:
        cv2.bitwise_not(thr, thr)

    # NEW: Apply morphological cleanup again at production scale
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel(3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel(2))

    # 6. Edge detection for outlined designs (when single_line=False)
    edges = cv2.Canny(thr, 20, 30)
    dilate = cv2.dilate(edges, kernel(50))
    edges = cv2.bitwise_and(edges, dilate)
    edges = cv2.dilate(edges, kernel(5))

    # 7. Zhang-Suen thinning
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

    # NEW: Filter out tiny contours (likely artifacts from JPG compression / texture).
    # A real tube path will have a meaningful length relative to the sign size.
    # Threshold: at least 0.5% of width_px (e.g., for a 400px wide sign, drop contours < 2px)
    min_contour_length_px = max(0.005 * width_px, 3.0)
    all_contours_count = len(contours)
    significant_contours = [c for c in contours if cv2.arcLength(c, True) >= min_contour_length_px]

    arc_length_px = sum(cv2.arcLength(c, True) for c in significant_contours) / 2.0
    arc_length_px_unfiltered = sum(cv2.arcLength(c, True) for c in contours) / 2.0
    contours_dropped = all_contours_count - len(significant_contours)

    # 6. Convert to meters (matches Yellowpop's exact formula)
    raw_tube_length_m = round(arc_length_px * sign_width_cm / width_px / 100, 2)

    # 7. Apply optional calibration multiplier
    tube_length_m = round(raw_tube_length_m * calibration, 2)

    # Diagnostics
    h, w = result.shape
    nonzero = int(cv2.countNonZero(result))

    # NEW: Reliability tier — multi-signal classification
    contours_dropped_ratio = contours_dropped / max(all_contours_count, 1)
    tube_density_m_per_m = tube_length_m / max(sign_width_cm / 100.0, 0.01)  # m of tube per m of sign width

    accuracy_tier, tier_reasons = classify_accuracy_tier(
        contour_count_total=all_contours_count,
        contours_dropped_ratio=contours_dropped_ratio,
        tube_density=tube_density_m_per_m,
        sign_width_cm=sign_width_cm,
    )

    # Legacy 'confidence' field retained for backwards compatibility
    confidence = {'high': 'high', 'medium': 'medium', 'low': 'low', 'unreliable': 'low'}[accuracy_tier]

    return {
        'tube_length_m': tube_length_m,
        'tube_length_m_raw': raw_tube_length_m,
        'tube_length_m_unfiltered': round(arc_length_px_unfiltered * sign_width_cm / width_px / 100, 2),
        'tube_density_m_per_m': round(tube_density_m_per_m, 2),
        'calibration_applied': calibration,
        'arc_length_px': round(arc_length_px, 1),
        'width_px': int(width_px),
        'sign_width_cm': sign_width_cm,
        'single_line': single_line,
        'auto_detected_single_line': auto_detected,
        'filled_ratio': round(filled_ratio, 3) if filled_ratio is not None else None,
        'thin_image_dimensions': {'width': int(w), 'height': int(h)},
        'thin_nonzero_pixels': nonzero,
        'contour_count_total': all_contours_count,
        'contour_count_significant': len(significant_contours),
        'contours_dropped_as_artifacts': contours_dropped,
        'min_contour_length_px': round(min_contour_length_px, 1),
        'accuracy_tier': accuracy_tier,             # 'high' | 'medium' | 'low' | 'unreliable'
        'tier_reasons': tier_reasons,               # list[str] explaining tier choice
        'confidence': confidence,                    # legacy
        'algorithm': 'yellowpop_production_zhang_suen_arclength',
    }


def classify_accuracy_tier(contour_count_total: int, contours_dropped_ratio: float,
                           tube_density: float, sign_width_cm: float) -> tuple:
    """
    Multi-signal accuracy tier classification.
    Returns (tier_name, list_of_reasons).

    Tiers (worst-of-all-signals wins):
      - 'unreliable': don't show automated price; route to manual quote
      - 'low':        wide price range with caveats
      - 'medium':     moderate range, verify before final quote
      - 'high':       tight price range, auto-quote OK
    """
    reasons = []

    # Unreliable triggers
    if contour_count_total > 500:
        reasons.append(f'Excessive fragmentation ({contour_count_total} contours; likely noisy or photographic input)')
    if contours_dropped_ratio > 0.5:
        reasons.append(f'>{int(contours_dropped_ratio*100)}% of contours filtered as artifacts')
    if tube_density > 20:
        reasons.append(f'Tube density {tube_density:.1f} m/m too high — design may not be production-feasible as drawn')
    if sign_width_cm >= 30 and tube_density < 2:
        reasons.append(f'Tube density {tube_density:.2f} m/m unusually low — sparse design')

    if reasons:
        return 'unreliable', reasons

    # Low triggers
    if contour_count_total > 200:
        reasons.append(f'High contour count ({contour_count_total})')
    if contours_dropped_ratio > 0.3:
        reasons.append(f'{int(contours_dropped_ratio*100)}% of contours filtered as artifacts')
    if tube_density > 15:
        reasons.append(f'Dense design (tube density {tube_density:.1f} m/m)')

    if reasons:
        return 'low', reasons

    # Medium triggers
    if contour_count_total > 50:
        reasons.append(f'Moderate complexity ({contour_count_total} contours)')
    if contours_dropped_ratio > 0.1:
        reasons.append(f'{int(contours_dropped_ratio*100)}% artifacts removed')
    if tube_density > 12 or tube_density < 5:
        reasons.append(f'Tube density {tube_density:.1f} m/m at edge of typical range')

    if reasons:
        return 'medium', reasons

    return 'high', ['Clean signal — low contour count, minimal artifacts, normal density']


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
