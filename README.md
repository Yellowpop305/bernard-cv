# Bernard CV — Tube Length Estimation Microservice

A small Python + Flask service that takes a design image URL and returns precise tube length estimation using computer vision (skeletonization).

Companion to the main `bernard-agent` Node service. Used to replace Claude's intuition-based tube length estimates with deterministic pixel measurement.

## How it works

```
POST /trace-tube
  body: { "image_url": "...", "sign_width_cm": 80 }

Pipeline:
  1. Download the image
  2. Convert to grayscale
  3. Auto-detect: line art on white BG vs. neon mockup on dark BG
  4. Threshold to binary (extract strokes from background)
  5. Light morphological cleanup (remove specks)
  6. Find content bounding box (where the design lives)
  7. Skeletonize (every stroke → 1-pixel-wide centerline)
  8. Count skeleton pixels
  9. Convert pixels → cm using sign_width_cm as scale
  10. Multiply by 1.1 (matches Yellowpop's tube formula)

Returns:
  {
    "tube_length_m": 7.42,         // ← what the pricing engine needs
    "skeleton_pixels": 7180,
    "stroke_length_cm": 674.5,
    "detected_type": "line_art",   // or "mockup"
    "confidence": "high",
    "image_dimensions": { "width": 800, "height": 800 },
    "content_bounds": { ... },
    "cm_per_pixel": 0.094,
    "multiplier_applied": 1.1
  }
```

## Tech stack

- Python 3.11
- Flask + gunicorn (web server)
- OpenCV (image processing)
- scikit-image (skeletonization — medial axis transform)
- Pillow + requests (image fetching)

## Local development

```bash
cd bernard-cv

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run locally
python app.py
# Service runs at http://localhost:8000
```

Test it:

```bash
curl -X POST http://localhost:8000/trace-tube \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://yellowpop.nyc3.digitaloceanspaces.com/995154/189472_e6834dbf5349ba9dce021ed18389c480.png",
    "sign_width_cm": 109
  }'
```

## Calibration test

Once running, validate against the 8 known calibration deals:

```bash
python test_calibration.py http://localhost:8000
# or against the deployed Railway version:
python test_calibration.py https://bernard-cv.up.railway.app
```

You'll get a table comparing predicted tube length vs actual for each deal. Target: ±10% error on most deals.

> Note: update `CALIBRATION_DEALS` in `test_calibration.py` with the real S3 URLs first.

## Railway deployment

Same pattern as bernard-agent:

1. Push this folder to GitHub repo (private): `Yellowpop305/bernard-cv`
2. Railway → New Project → Deploy from GitHub repo → select `bernard-cv`
3. Railway auto-detects Python from `requirements.txt` + `Procfile`
4. Wait for build (takes 2-3 min — OpenCV is bigger than typical Python deps)
5. Settings → Public Networking → Generate Domain (port 8000 or auto)

You'll get a URL like `https://bernard-cv-production.up.railway.app`.

Then in **bernard-agent's** Railway env vars, add:

```
BERNARD_CV_URL=https://bernard-cv-production.up.railway.app
```

Bernard will then call this service for tube length estimation on every deal with an asset.

## Cost estimate

- Railway compute: ~$5-10/month for a small Python service at 5,000 leads/month
- No Anthropic / API costs (this is pure CPU)
- Image bandwidth: negligible (yellowpop.nyc3.digitaloceanspaces.com → Railway, < 5 MB per image)

Total marginal cost: **~$10/month** to add ~5x improvement in tube length accuracy.

## Edge cases

| Image type | How it's handled |
|---|---|
| Clean line art (white BG, dark strokes) | Adaptive threshold → skeletonize. Best accuracy. |
| Neon mockup (dark BG, glowing white strokes) | Auto-detected via brightness, kept bright pixels |
| Outline + filled shapes (double-line) | Skeletonize collapses both edges into one centerline ✓ |
| Photographic backgrounds | Confidence flagged "low" — Bernard falls back to vision estimate |
| Very dense detail | Skeleton might over-count if strokes touch. Validated with calibration set. |

## Future improvements

- Machine learning for "is this a tube vs decoration?" classification
- Per-color separation for multi-color designs (track separate tube runs per color)
- Sectioning detection (multi-piece signs)
- Output an annotated debug image showing the detected skeleton overlay
