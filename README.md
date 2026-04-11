# AR Image Stitcher

A web application for stitching multiple AR/aerial images into a seamless panorama. Upload images in sequence, reorder them if needed, and get a single stitched output.

## Setup

```bash
# Create and activate virtual environment (Python 3.12)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install flask opencv-python numpy pillow tensorflow keras matplotlib
```

## Running the Web App

```bash
source venv/bin/activate
python server.py
# Open http://localhost:5001
```

## How It Works

1. Upload 2 or more images using the drag-and-drop zone
2. Thumbnails appear in numbered order — drag to reorder if needed
3. Click **Stitch Images** — a loading overlay shows while processing
4. The stitched panorama appears with width/height info and a download button

## Stitching Pipeline (`server.py`)

Images are stitched left → right in the order shown in the UI:

1. **First pair** — `cv2.Stitcher` (PANORAMA or SCANS mode) produces a high-quality base panorama with bundle adjustment, cylindrical projection, seam-finding, multi-band blending, and exposure compensation.
2. **Each additional image** — SIFT feature detection (8000 keypoints) + Lowe's ratio test (0.8) + RANSAC homography maps the new image into the panorama's coordinate frame, then distance-weighted feather blending merges the overlap.
3. **Fallback** — if SIFT matching fails for any pair, images are concatenated side-by-side so no image is ever dropped.

Black borders introduced by perspective warping are cropped from the final result.

## Other Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `app.py` | Trains a Pix2Pix GAN on overlapping image pairs | `gan_stitched.png` |
| `feature_extractor.py` | SIFT/ORB/AKAZE feature extraction and matching | `./feature_output/` |
| `pose_estimator.py` | Camera pose estimation + 3D triangulation | `./pose_output/` |

```bash
# Run standalone scripts (all read from ./images/)
python app.py
python feature_extractor.py
python pose_estimator.py
```

## Input / Output

- **Input:** JPG, PNG, BMP — placed in `./images/` for the CLI scripts, or uploaded via the web UI
- **Output:** Stitched JPEG panorama (downloadable from the UI), or `gan_stitched.png` / `./feature_output/` / `./pose_output/` for the CLI scripts
- **Point cloud** (from `pose_estimator.py`): ASCII PLY format, viewable in MeshLab or CloudCompare
