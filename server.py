from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

MAX_DIM = 1500   # Resize images to this max dimension before stitching
MAX_OUT_W = 12000
MAX_OUT_H = 6000


def load_image(file_bytes: bytes) -> np.ndarray:
    """Load image from bytes, auto-correcting EXIF orientation."""
    pil = Image.open(io.BytesIO(file_bytes))
    pil = ImageOps.exif_transpose(pil).convert('RGB')
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def stitch_images(images: list) -> np.ndarray:
    """
    Stitch images in provided order (left → right).

    1. cv2.Stitcher for the first pair  (high-quality base panorama).
    2. Each subsequent image is added via SIFT + homography + distance-blend.
    """
    if len(images) == 1:
        return images[0]

    resized = []
    for img in images:
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        resized.append(img)

    # ── Step 1: stitch first pair with cv2.Stitcher ───────────────────────
    result = _opencv_pair(resized[0], resized[1])
    if result is None:
        # Fallback: manual for the first pair too
        sift = cv2.SIFT_create(nfeatures=8000)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        H = _find_homography(resized[0], resized[1], sift, bf)
        if H is not None:
            result = _warp_and_blend(resized[0], resized[1], H)
        else:
            result = _side_by_side(resized[0], resized[1])

    # ── Step 2: add each remaining image via SIFT + homography ────────────
    sift = cv2.SIFT_create(nfeatures=8000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    for i in range(2, len(resized)):
        H = _find_homography(result, resized[i], sift, bf)
        if H is not None:
            result = _warp_and_blend(result, resized[i], H)
        else:
            result = _side_by_side(result, resized[i])

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _opencv_pair(a, b):
    """Stitch exactly two original images with cv2.Stitcher."""
    for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
        try:
            stitcher = cv2.Stitcher_create(mode)
            stitcher.setPanoConfidenceThresh(0.3)
            status, pano = stitcher.stitch([a, b])
            if status == cv2.Stitcher_OK:
                return pano
        except Exception:
            continue
    return None


def _side_by_side(a, b):
    """Height-normalised horizontal concatenation."""
    ha, hb = a.shape[0], b.shape[0]
    if ha != hb:
        b = cv2.resize(b, (int(b.shape[1] * ha / hb), ha))
    return np.hstack([a, b])


def _warp_and_blend(img_left, img_right, H):
    """Warp img_right onto img_left's frame and distance-blend the seam."""
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    corners_r = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners_l = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped_c = cv2.perspectiveTransform(corners_r, H)
    all_pts = np.concatenate([corners_l, warped_c])

    x0 = int(np.floor(all_pts[:, :, 0].min())) - 2
    y0 = int(np.floor(all_pts[:, :, 1].min())) - 2
    x1 = int(np.ceil(all_pts[:, :, 0].max())) + 2
    y1 = int(np.ceil(all_pts[:, :, 1].max())) + 2

    dx, dy = -x0, -y0
    out_w = min(x1 - x0, MAX_OUT_W)
    out_h = min(y1 - y0, MAX_OUT_H)
    T = np.float64([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    warped = cv2.warpPerspective(img_right, T @ H, (out_w, out_h))
    mask_r = cv2.warpPerspective(np.ones((h2, w2), np.uint8) * 255,
                                  T @ H, (out_w, out_h))

    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    mask_l = np.zeros((out_h, out_w), np.uint8)
    ye = min(dy + h1, out_h)
    xe = min(dx + w1, out_w)
    canvas[dy:ye, dx:xe] = img_left[:ye - dy, :xe - dx]
    mask_l[dy:ye, dx:xe] = 255

    d_l = cv2.distanceTransform(mask_l, cv2.DIST_L2, 5).astype(np.float32)
    d_r = cv2.distanceTransform(mask_r, cv2.DIST_L2, 5).astype(np.float32)
    ws = np.maximum(d_l + d_r, 1e-6)

    result = (canvas.astype(np.float32) * (d_l / ws)[:, :, None] +
              warped.astype(np.float32) * (d_r / ws)[:, :, None])

    result = np.clip(result, 0, 255).astype(np.uint8)
    return _crop_black(result)


def _find_homography(img1, img2, sift, bf):
    """Return H that maps img2 → img1 coords, or None on failure."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None

    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.8 * n.distance]
    if len(good) < 10:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None or mask is None or int(mask.sum()) < 8:
        return None
    return H


def _blend_distance(warped_imgs, warped_masks, out_h, out_w):
    """Feather-blend using distance-from-edge weights (smooth seams)."""
    weight_maps = []
    for mask in warped_masks:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        weight_maps.append(dist.astype(np.float32))

    weight_sum = sum(weight_maps)
    weight_sum = np.maximum(weight_sum, 1e-6)

    result = np.zeros((out_h, out_w, 3), np.float32)
    for img, wt in zip(warped_imgs, weight_maps):
        alpha = (wt / weight_sum)[:, :, np.newaxis]
        result += img.astype(np.float32) * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def _crop_black(img):
    """Simple bounding-rect crop of non-black content."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stitch', methods=['POST'])
def api_stitch():
    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'error': 'Upload at least 2 images'}), 400

    images = []
    for f in files:
        try:
            img = load_image(f.read())
        except Exception as e:
            return jsonify({'error': f'Could not decode "{f.filename}": {e}'}), 400
        images.append(img)

    try:
        result = stitch_images(images)
    except Exception as e:
        return jsonify({'error': f'Stitching failed: {e}'}), 500

    _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode()
    h, w = result.shape[:2]
    return jsonify({'image': b64, 'width': w, 'height': h})


if __name__ == '__main__':
    print("Starting AR Image Stitcher at http://localhost:5000")
    app.run(debug=True, port=5001)
