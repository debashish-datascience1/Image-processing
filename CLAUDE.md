# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Activate virtual environment (Python 3.12.7)
source venv/bin/activate

# GAN-based image stitching (trains a Pix2Pix GAN, outputs gan_stitched.png)
python app.py

# Feature extraction and matching (outputs to ./feature_output/)
python feature_extractor.py

# Camera pose estimation + 3D point cloud (outputs to ./pose_output/)
python pose_estimator.py
```

All scripts expect input images in `./images/`. Outputs go to `./feature_output/`, `./pose_output/`, or the project root.

## Architecture

Three independent pipelines for image stitching and 3D reconstruction:

### 1. GAN-based stitching (`app.py`)
`DeepGANImageStitcher` trains a Pix2Pix-style GAN on overlapping image pairs:
- **Generator**: U-Net (4-level encoder → bottleneck → 4-level decoder with skip connections)
- **Discriminator**: PatchGAN classifier
- **Loss**: Adversarial (BCE) + L1 reconstruction
- Images are resized to 256px height, normalized to [-1, 1]
- Overlap region uses alpha blending to create training pairs

### 2. Feature-based pipeline (`feature_extractor.py` → `pose_estimator.py`)
`FeatureExtractor` → `PoseEstimator` form a sequential 3D reconstruction pipeline:

1. **Feature extraction**: SIFT (default, 5000 features), ORB, or AKAZE
2. **Matching**: k-NN + Lowe's ratio test (threshold: 0.75) + RANSAC (5px)
3. **Pose estimation**: Camera intrinsics from FOV (default 60°); essential matrix → R, t via `cv2.recoverPose()`
4. **Triangulation**: `cv2.triangulatePoints()` → filters points with Z ∈ (0, 100)
5. **Output**: Camera poses 3D plot + ASCII PLY point cloud

`pose_estimator.py` imports `FeatureExtractor` from `feature_extractor.py`.

## Key Parameters (all hardcoded)

| File | Parameter | Value |
|------|-----------|-------|
| `app.py` | Image height | 256 px |
| `app.py` | Training epochs | 500 |
| `app.py` | Learning rate | 0.0002 |
| `feature_extractor.py` | Match ratio threshold | 0.75 |
| `feature_extractor.py` | RANSAC threshold | 5.0 px |
| `pose_estimator.py` | Default FOV | 60° |
| `pose_estimator.py` | Depth filter range | Z ∈ (0, 100) |

## Dependencies

- **TensorFlow 2.20 / Keras 3.12** — GAN training (`app.py`)
- **OpenCV 4.12** — Feature detection, pose estimation, triangulation
- **NumPy 2.2**, **Pillow 12**, **Matplotlib 3.10** — Array ops, image I/O, visualization

The PLY point cloud output can be viewed in MeshLab or CloudCompare.
