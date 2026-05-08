"""
Bernard CV — Computer Vision microservice for tube length estimation.
v3: skeletonize + light branch pruning + production calibration multiplier.

Pipeline:
  1. Download image, auto-detect type (line_art vs mockup)
  2. Threshold to binary
  3. Skeletonize to get medial axis (no erosion — was overcorrecting)
  4. Light branch pruning via skan (drop branches < 1% of longest — removes only obvious artifacts)
  5. Convert pixels → cm using sign_width as scale
  6. Apply 1.1x tube multiplier (Yellowpop's formula)
  7. Apply calibration multiplier (designer optimization factor) — tunable via env var
"""

from flask import Flask, request, jsonify
import requests
from io import BytesIO
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from PIL import Image
import os

app = Flask(__name__)

# Yellowpop's standard tube thickness (informational; passed through but not used to erode)
DEFAULT_TUBE_THICKNESS_MM = 8.0

# Branches shorter than this fraction of the longest branch are pruned (likely artifacts)
# 1% threshold = removes tiny serif/junction noise but keeps real tube paths
BRANCH_PRUNE_THRESHOLD = float(os.environ.get('BRANCH_PRUNE_THRESHOLD', '0.01'))

# Empirical calibration multiplier — accounts for the gap between literal skeleton length
# and actual production tube length (designers optimize routing, skip decorative micro-details).
# Tuned against Yellowpop's calibration set. Adjust as more data comes in.
PRODUCTION_CALIBRATION = float(os.environ.get('PRODUCTION_CALIBRATION', '0.85'))


# ---------- Health & info endpoints ----------

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'bernard-cv',
        'status': 'ok',
        'version': '3.0.0',
        'config': {
            'branch_prune_threshold': BRANCH_PRUNE_THRESHOLD,
            'production_calibration': PRODUCTION_CALIBRATION,
        },
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
    data = request.get_json(silent=True) or {}
    image_url = data.get('image_url')
    sign_width_cm = float(data.get('sign_width_cm', 80))
    image_type = data.get('image_type', 'auto')
    # Allow override per request (otherwise use the env-driven default)
    calibration = float(data.get('calibration', PRODUCTION_CALIBRATION))

    if not image_url:
        return jsonify({'error': 'Missing image_url'}), 400

    try:
        result = analyze_image(image_url, sign_width_cm, image_type, calibration)
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({'error': f'Failed to fetch image: {e}'}), 400
    except Exception as e:
        app.logger.exception('Processing failed')
        return jsonify({'error': f'Processing failed: {e}'}), 500


# ---------- Image analysis pipeline ----------

def analyze_image(image_url: str, sign_width_cm: float, image_type: str, calibration: float) -> dict:
    # 1. Download & load
    response = requests.get(image_url, timeout=20)
    response.raise_for_status()
    pil_img = Image.open(BytesIO(response.content)).convert('RGB')
    img = np.array(pil_img)
    h, w = img.shape[:2]

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. Detect image type (auto)
    avg_brightness = float(np.mean(gray))
    detected_type = image_type
    if image_type == 'auto':
        detected_type = 'mockup' if avg_brightness < 110 else 'line_art'

    # 4. Threshold to binary
    if detected_type == 'mockup':
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    elif detected_type == 'line_art':
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
    else:
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # 5. Light morphological cleanup — remove tiny specks
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Find content bounding box (used for cm/pixel scaling)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return {
            'error': 'No design content detected',
            'image_dimensions': {'width': w, 'height': h},
            'detected_type': detected_type,
            'avg_brightness': avg_brightness,
        }
    x, y, content_w, content_h = cv2.boundingRect(coords)

    # 7. Skeletonize the binary mask directly (no erosion)
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    raw_skeleton_pixels = int(np.sum(skeleton))

    # 8. Light branch pruning via skan — drop only obvious artifacts (< 1% of longest branch)
    pruned_pixels = float(raw_skeleton_pixels)  # default fallback
    branch_count_total = 0
    branch_count_kept = 0
    pruned_pixels_removed = 0

    if raw_skeleton_pixels > 50:
        try:
            skel_obj = Skeleton(skeleton)
            df = summarize(skel_obj)
            if len(df) > 0:
                max_branch = float(df['branch-distance'].max())
                total = float(df['branch-distance'].sum())
                # Threshold: 1% of longest branch, but at least 5 pixels (handles small images)
                min_keep = max(BRANCH_PRUNE_THRESHOLD * max_branch, 5.0)

                kept = df[df['branch-distance'] >= min_keep]
                branch_count_total = int(len(df))
                branch_count_kept = int(len(kept))
                kept_length = float(kept['branch-distance'].sum())
                pruned_pixels_removed = int(round(total - kept_length))
                pruned_pixels = kept_length
        except Exception as e:
            app.logger.warning(f'skan branch pruning failed: {e}')

    # 9. Pixel → cm using sign_width as scale (use content_w, not image w, for accurate scale)
    cm_per_pixel = sign_width_cm / max(content_w, 1)
    stroke_length_cm = pruned_pixels * cm_per_pixel

    # 10. Apply tube multiplier (1.1x — matches Yellowpop's getProductionPrice formula)
    tube_length_m_raw = (stroke_length_cm / 100.0) * 1.1

    # 11. Apply production calibration multiplier
    # (literal skeleton ≠ actual production tube, designers optimize)
    tube_length_m = tube_length_m_raw * calibration

    # 12. Confidence
    content_ratio = (content_w * content_h) / (w * h)
    if content_ratio > 0.3 and pruned_pixels > 200:
        confidence = 'high'
    elif content_ratio > 0.1:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'tube_length_m': round(tube_length_m, 2),
        'tube_length_m_raw_skeleton': round(tube_length_m_raw, 2),
        'stroke_length_cm': round(stroke_length_cm, 2),
        'skeleton_pixels': int(round(pruned_pixels)),
        'raw_skeleton_pixels_before_pruning': raw_skeleton_pixels,
        'pruned_pixels_removed': pruned_pixels_removed,
        'branch_count_total': branch_count_total,
        'branch_count_kept': branch_count_kept,
        'image_dimensions': {'width': w, 'height': h},
        'content_bounds': {'x': int(x), 'y': int(y), 'width': int(content_w), 'height': int(content_h)},
        'content_ratio': round(content_ratio, 3),
        'detected_type': detected_type,
        'avg_brightness': round(avg_brightness, 1),
        'cm_per_pixel': round(cm_per_pixel, 4),
        'confidence': confidence,
        'multiplier_applied': 1.1,
        'production_calibration': calibration,
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
