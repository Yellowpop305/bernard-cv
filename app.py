"""
Bernard CV — Computer Vision microservice for tube length estimation.

Takes a design image URL + sign width in cm, returns precise tube length
by skeletonizing the line art and counting centerline pixels.

Companion service to bernard-agent. Lives at https://bernard-cv.up.railway.app/
"""

from flask import Flask, request, jsonify
import requests
from io import BytesIO
import numpy as np
import cv2
from skimage.morphology import skeletonize
from PIL import Image
import os

app = Flask(__name__)

# ---------- Health & info endpoints ----------

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'bernard-cv',
        'status': 'ok',
        'version': '1.0.0',
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
    Body: { "image_url": "https://...", "sign_width_cm": 80, "image_type"?: "auto|line_art|mockup" }
    Returns JSON with tube_length_m and diagnostics.
    """
    data = request.get_json(silent=True) or {}
    image_url = data.get('image_url')
    sign_width_cm = float(data.get('sign_width_cm', 80))
    image_type = data.get('image_type', 'auto')

    if not image_url:
        return jsonify({'error': 'Missing image_url'}), 400

    try:
        result = analyze_image(image_url, sign_width_cm, image_type)
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({'error': f'Failed to fetch image: {e}'}), 400
    except Exception as e:
        app.logger.exception('Processing failed')
        return jsonify({'error': f'Processing failed: {e}'}), 500


# ---------- Image analysis pipeline ----------

def analyze_image(image_url: str, sign_width_cm: float, image_type: str = 'auto') -> dict:
    # 1. Download
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
        # Heuristic: if avg brightness < 100, it's a dark-bg mockup
        # if > 200, it's clean line art on white
        if avg_brightness < 110:
            detected_type = 'mockup'
        else:
            detected_type = 'line_art'

    # 4. Threshold to binary based on type
    if detected_type == 'mockup':
        # Dark background, bright glow lines — keep brightest pixels
        # Use Otsu on inverted to handle varying brightness
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    elif detected_type == 'line_art':
        # Light background, dark strokes — invert (we want strokes = 1)
        # Adaptive threshold handles uneven backgrounds well
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
    else:
        # Fallback simple threshold
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # 5. Light morphological cleanup — remove tiny specks (noise) but preserve thin strokes
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Find content bounding box (where the design actually lives)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return {
            'error': 'No design content detected',
            'image_dimensions': {'width': w, 'height': h},
            'detected_type': detected_type,
            'avg_brightness': avg_brightness,
        }
    x, y, content_w, content_h = cv2.boundingRect(coords)

    # 7. Skeletonize — extract 1-pixel-wide centerlines
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    skeleton_pixel_count = int(np.sum(skeleton))

    # 8. Convert pixels to cm using sign width as scale reference
    # Use content_width (not image width) since the design typically has padding
    cm_per_pixel = sign_width_cm / max(content_w, 1)
    stroke_length_cm = skeleton_pixel_count * cm_per_pixel

    # 9. Apply tube multiplier (matches Yellowpop's getProductionPrice formula: tubeLength * 1.1)
    tube_length_m = (stroke_length_cm / 100.0) * 1.1

    # 10. Confidence: high if content is well-defined and reasonably large
    content_ratio = (content_w * content_h) / (w * h)
    if content_ratio > 0.3 and skeleton_pixel_count > 200:
        confidence = 'high'
    elif content_ratio > 0.1:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'tube_length_m': round(tube_length_m, 2),
        'stroke_length_cm': round(stroke_length_cm, 2),
        'skeleton_pixels': skeleton_pixel_count,
        'image_dimensions': {'width': w, 'height': h},
        'content_bounds': {'x': int(x), 'y': int(y), 'width': int(content_w), 'height': int(content_h)},
        'content_ratio': round(content_ratio, 3),
        'detected_type': detected_type,
        'avg_brightness': round(avg_brightness, 1),
        'cm_per_pixel': round(cm_per_pixel, 4),
        'confidence': confidence,
        'multiplier_applied': 1.1,
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
