"""
Bernard CV — Computer Vision microservice for tube length estimation.

Takes a design image URL + sign width in cm, returns precise tube length
by skeletonizing the line art and counting centerline pixels, with:
  - Tube-thickness-aware preprocessing (erode by half tube thickness)
  - Branch pruning (drop spurious short branches at junctions/serifs)

Companion service to bernard-agent. Lives at https://bernard-cv-production.up.railway.app/
"""

from flask import Flask, request, jsonify
import requests
from io import BytesIO
import numpy as np
import cv2
from skimage.morphology import skeletonize, binary_erosion, disk
from skan import Skeleton, summarize
from PIL import Image
import os

app = Flask(__name__)

# Yellowpop's standard tube thickness
DEFAULT_TUBE_THICKNESS_MM = 8.0

# Branches shorter than this fraction of total skeleton are pruned (likely artifacts)
BRANCH_PRUNE_THRESHOLD = 0.03  # 3% — drops serifs / tiny junction branches


# ---------- Health & info endpoints ----------

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'bernard-cv',
        'status': 'ok',
        'version': '2.0.0',
        'features': {
            'auto_image_type_detection': True,
            'tube_thickness_aware': True,
            'branch_pruning': True,
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
    """
    POST /trace-tube
    Body: {
        "image_url": "https://...",
        "sign_width_cm": 80,
        "tube_thickness_mm"?: 8,
        "image_type"?: "auto|line_art|mockup"
    }
    Returns JSON with tube_length_m and diagnostics.
    """
    data = request.get_json(silent=True) or {}
    image_url = data.get('image_url')
    sign_width_cm = float(data.get('sign_width_cm', 80))
    tube_thickness_mm = float(data.get('tube_thickness_mm', DEFAULT_TUBE_THICKNESS_MM))
    image_type = data.get('image_type', 'auto')

    if not image_url:
        return jsonify({'error': 'Missing image_url'}), 400

    try:
        result = analyze_image(image_url, sign_width_cm, tube_thickness_mm, image_type)
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({'error': f'Failed to fetch image: {e}'}), 400
    except Exception as e:
        app.logger.exception('Processing failed')
        return jsonify({'error': f'Processing failed: {e}'}), 500


# ---------- Image analysis pipeline ----------

def analyze_image(image_url: str, sign_width_cm: float, tube_thickness_mm: float, image_type: str = 'auto') -> dict:
    # 1. Download & load
    response = requests.get(image_url, timeout=20)
    response.raise_for_status()
    raw = response.content
    pil_img = Image.open(BytesIO(raw)).convert('RGB')
    img = np.array(pil_img)
    h, w = img.shape[:2]

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. Detect image type if auto
    avg_brightness = float(np.mean(gray))
    detected_type = image_type
    if image_type == 'auto':
        if avg_brightness < 110:
            detected_type = 'mockup'
        else:
            detected_type = 'line_art'

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

    # 6. Find content bounding box
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return {
            'error': 'No design content detected',
            'image_dimensions': {'width': w, 'height': h},
            'detected_type': detected_type,
            'avg_brightness': avg_brightness,
        }
    x, y, content_w, content_h = cv2.boundingRect(coords)

    # 7. NEW: Tube-thickness-aware erosion
    # Convert 8mm tube thickness to pixel radius using sign width as scale.
    # Erode by HALF the tube thickness — this collapses bold/filled strokes
    # to a thinner skeleton-friendly shape, reducing spurious branches.
    cm_per_pixel = sign_width_cm / max(content_w, 1)
    tube_thickness_cm = tube_thickness_mm / 10.0
    erosion_radius_px = max(1, int(round(0.5 * tube_thickness_cm / cm_per_pixel)))

    # Only erode if strokes are thick enough to benefit (avoid eroding line art away)
    binary_bool = binary > 0
    binary_eroded = binary_bool
    if erosion_radius_px > 1:
        # Check if erosion would obliterate too much content
        eroded_test = binary_erosion(binary_bool, disk(erosion_radius_px))
        if np.sum(eroded_test) >= 0.3 * np.sum(binary_bool):
            # Safe to erode — preserves at least 30% of strokes
            binary_eroded = eroded_test
        else:
            # Strokes are already thin (line art) — skip erosion
            erosion_radius_px = 0

    # 8. Skeletonize
    skeleton = skeletonize(binary_eroded)
    raw_skeleton_pixels = int(np.sum(skeleton))

    # 9. NEW: Branch pruning via skan
    # Enumerate skeleton branches; drop ones shorter than threshold
    pruned_pixels = raw_skeleton_pixels
    branch_count_total = 0
    branch_count_kept = 0
    branches_pruned_pixels = 0

    if raw_skeleton_pixels > 50:
        try:
            skel_obj = Skeleton(skeleton)
            df = summarize(skel_obj)
            if len(df) > 0:
                # Total length of all branches
                total_branch_length = df['branch-distance'].sum()
                # Threshold: keep branches >= 3% of the LONGEST branch
                # (anything tinier is almost certainly a serif/junction artifact)
                max_branch = df['branch-distance'].max()
                min_keep_length = BRANCH_PRUNE_THRESHOLD * max_branch
                # Also keep branches > 5 pixels regardless (handle very small images)
                min_keep_length = max(min_keep_length, 5.0)

                kept = df[df['branch-distance'] >= min_keep_length]
                pruned = df[df['branch-distance'] < min_keep_length]

                branch_count_total = int(len(df))
                branch_count_kept = int(len(kept))
                kept_length = float(kept['branch-distance'].sum())
                branches_pruned_pixels = int(round(total_branch_length - kept_length))
                pruned_pixels = int(round(kept_length))
        except Exception as e:
            # If skan fails (rare), fall back to raw skeleton
            app.logger.warning(f'skan branch pruning failed: {e}')

    # 10. Convert pixels → cm using sign_width as scale
    stroke_length_cm = pruned_pixels * cm_per_pixel

    # 11. Apply tube multiplier (matches Yellowpop's getProductionPrice formula: tubeLength * 1.1)
    tube_length_m = (stroke_length_cm / 100.0) * 1.1

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
        'stroke_length_cm': round(stroke_length_cm, 2),
        'skeleton_pixels': pruned_pixels,
        'raw_skeleton_pixels_before_pruning': raw_skeleton_pixels,
        'pruned_pixels_removed': branches_pruned_pixels,
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
        'tube_thickness_mm': tube_thickness_mm,
        'erosion_radius_px': erosion_radius_px,
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
