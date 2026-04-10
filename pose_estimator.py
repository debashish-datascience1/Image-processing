import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict, Optional
import os


class PoseEstimator:
    """
    Estimate camera poses from matched feature points.
    This converts 2D image matches into 3D camera positions and orientations.
    """
    
    def __init__(self, focal_length: float = None, principal_point: Tuple = None,
                 image_size: Tuple = None):
        """
        Initialize pose estimator.
        
        Args:
            focal_length: Camera focal length in pixels (if known)
            principal_point: (cx, cy) principal point (if known)
            image_size: (width, height) of images
        """
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.K = None  # Camera intrinsic matrix
        
    def estimate_camera_matrix(self, image_shape: Tuple, 
                               fov_degrees: float = 60.0) -> np.ndarray:
        """
        Estimate camera intrinsic matrix from image size.
        
        Args:
            image_shape: (height, width, channels) or (height, width)
            fov_degrees: Field of view in degrees (typical: 50-70)
            
        Returns:
            K: 3x3 camera intrinsic matrix
        """
        h, w = image_shape[:2]
        
        # Estimate focal length from FOV
        if self.focal_length is None:
            fov_rad = np.deg2rad(fov_degrees)
            focal = w / (2 * np.tan(fov_rad / 2))
        else:
            focal = self.focal_length
        
        # Principal point at image center
        if self.principal_point is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = self.principal_point
        
        # Build intrinsic matrix
        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.K = K
        print(f"\nCamera Intrinsic Matrix (K):")
        print(f"  Focal length: {focal:.2f} pixels")
        print(f"  Principal point: ({cx:.2f}, {cy:.2f})")
        print(f"  Image size: {w}x{h}")
        
        return K
    
    def decompose_essential_matrix(self, E: np.ndarray, pts1: np.ndarray, 
                                   pts2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose essential matrix to get rotation and translation.
        
        Args:
            E: Essential matrix
            pts1: Points from first image (Nx2)
            pts2: Points from second image (Nx2)
            K: Camera intrinsic matrix
            
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        
        print(f"\nCamera Pose Recovered:")
        print(f"  Rotation matrix R:")
        print(f"    {R[0]}")
        print(f"    {R[1]}")
        print(f"    {R[2]}")
        print(f"  Translation t: {t.T}")
        print(f"  Inliers used: {mask.sum()}/{len(pts1)}")
        
        return R, t
    
    def estimate_pose_from_fundamental(self, F: np.ndarray, pts1: np.ndarray,
                                      pts2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate camera pose from fundamental matrix.
        
        Args:
            F: Fundamental matrix
            pts1: Points from first image
            pts2: Points from second image
            K: Camera intrinsic matrix
            
        Returns:
            R: Rotation matrix
            t: Translation vector
        """
        # Convert fundamental to essential matrix: E = K^T * F * K
        E = K.T @ F @ K
        
        # Decompose essential matrix
        R, t = self.decompose_essential_matrix(E, pts1, pts2, K)
        
        return R, t
    
    def estimate_pose_from_matches(self, pts1: np.ndarray, pts2: np.ndarray,
                                   image_shape: Tuple, method: str = 'essential') -> Dict:
        """
        Estimate camera pose from matched points.
        
        Args:
            pts1: Matched points from first image (Nx2)
            pts2: Matched points from second image (Nx2)
            image_shape: Shape of images
            method: 'essential' or 'fundamental'
            
        Returns:
            Dictionary with R, t, K, E/F matrices
        """
        print("\n" + "="*60)
        print("CAMERA POSE ESTIMATION")
        print("="*60)
        
        # Estimate camera intrinsic matrix
        K = self.estimate_camera_matrix(image_shape)
        
        if method == 'essential':
            print(f"\n[1/2] Computing Essential Matrix...")
            E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
            print(f"  Essential matrix computed")
            print(f"  Inliers: {mask.sum()}/{len(pts1)}")
            
            print(f"\n[2/2] Recovering Camera Pose...")
            R, t = self.decompose_essential_matrix(E, pts1, pts2, K)
            
            return {
                'R': R,
                't': t,
                'K': K,
                'E': E,
                'method': 'essential',
                'inlier_mask': mask
            }
        
        else:  # fundamental
            print(f"\n[1/2] Computing Fundamental Matrix...")
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
            print(f"  Fundamental matrix computed")
            print(f"  Inliers: {mask.sum()}/{len(pts1)}")
            
            print(f"\n[2/2] Recovering Camera Pose...")
            R, t = self.estimate_pose_from_fundamental(F, pts1, pts2, K)
            
            return {
                'R': R,
                't': t,
                'K': K,
                'F': F,
                'method': 'fundamental',
                'inlier_mask': mask
            }
    
    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                          K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from 2D matches and camera poses.
        
        Args:
            pts1: Points from first image (Nx2)
            pts2: Points from second image (Nx2)
            K: Camera intrinsic matrix
            R: Rotation from camera 1 to camera 2
            t: Translation from camera 1 to camera 2
            
        Returns:
            points_3d: Nx3 array of 3D points
        """
        print("\n" + "="*60)
        print("3D POINT TRIANGULATION")
        print("="*60)
        
        # Projection matrices
        # Camera 1 at origin: P1 = K * [I | 0]
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # Camera 2 with pose: P2 = K * [R | t]
        P2 = K @ np.hstack([R, t])
        
        print(f"\nProjection Matrix P1 (Camera 1 at origin):")
        print(P1)
        print(f"\nProjection Matrix P2 (Camera 2):")
        print(P2)
        
        # Triangulate
        print(f"\nTriangulating {len(pts1)} points...")
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        
        # Filter points behind camera or too far
        valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 100)
        points_3d_filtered = points_3d[valid_mask]
        
        print(f"  Triangulated: {len(points_3d)} points")
        print(f"  Valid points (in front of camera): {len(points_3d_filtered)}")
        print(f"  Point cloud bounds:")
        print(f"    X: [{points_3d_filtered[:, 0].min():.2f}, {points_3d_filtered[:, 0].max():.2f}]")
        print(f"    Y: [{points_3d_filtered[:, 1].min():.2f}, {points_3d_filtered[:, 1].max():.2f}]")
        print(f"    Z: [{points_3d_filtered[:, 2].min():.2f}, {points_3d_filtered[:, 2].max():.2f}]")
        
        return points_3d_filtered
    
    def visualize_camera_poses(self, poses: List[Dict], points_3d: np.ndarray = None,
                              save_path: str = None):
        """
        Visualize camera poses and 3D points.
        
        Args:
            poses: List of pose dictionaries with 'R' and 't'
            points_3d: Optional 3D points to visualize
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        if points_3d is not None and len(points_3d) > 0:
            # Subsample for visualization
            n_points = min(len(points_3d), 5000)
            indices = np.random.choice(len(points_3d), n_points, replace=False)
            pts = points_3d[indices]
            
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                      c=pts[:, 2], cmap='viridis', s=1, alpha=0.5)
        
        # Plot camera positions and orientations
        for i, pose in enumerate(poses):
            R = pose['R']
            t = pose['t'].flatten()
            
            # Camera position
            cam_pos = -R.T @ t
            ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], 
                      c='red', s=100, marker='o', label=f'Camera {i+1}')
            
            # Camera orientation (optical axis)
            optical_axis = R.T @ np.array([0, 0, 1])
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                     optical_axis[0], optical_axis[1], optical_axis[2],
                     length=0.5, color='red', arrow_length_ratio=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses and 3D Point Cloud')
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.array([
            points_3d[:, 0].max() - points_3d[:, 0].min() if points_3d is not None else 1,
            points_3d[:, 1].max() - points_3d[:, 1].min() if points_3d is not None else 1,
            points_3d[:, 2].max() - points_3d[:, 2].min() if points_3d is not None else 1
        ]).max() / 2.0
        
        if points_3d is not None:
            mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
            mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
            mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
        else:
            mid_x = mid_y = mid_z = 0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  Saved pose visualization: {save_path}")
        
        plt.show()
    
    def save_point_cloud(self, points_3d: np.ndarray, filename: str = "point_cloud.ply"):
        """
        Save 3D points as PLY file (can be opened in MeshLab, CloudCompare, etc.)
        
        Args:
            points_3d: Nx3 array of 3D points
            filename: Output filename
        """
        with open(filename, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Write points
            for pt in points_3d:
                f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
        
        print(f"\n  Saved point cloud: {filename}")
        print(f"  Open with MeshLab, CloudCompare, or Open3D")


def demo_pose_estimation():
    """
    Demo: Estimate camera poses from feature matches.
    """
    from feature_extractor import FeatureExtractor
    
    print("\n" + "="*60)
    print("POSE ESTIMATION DEMO")
    print("="*60)
    
    # Check if images exist
    img_dir = "./images"
    if not os.path.exists(img_dir):
        print(f"\nError: {img_dir} directory not found!")
        print("Please run feature_extractor.py first.")
        return
    
    # Get image files
    image_files = sorted([f for f in os.listdir(img_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if len(image_files) < 2:
        print(f"\nError: Need at least 2 images in {img_dir}")
        return
    
    # Load images
    img1_path = os.path.join(img_dir, image_files[0])
    img2_path = os.path.join(img_dir, image_files[1])
    
    print(f"\nLoading images:")
    print(f"  Image 1: {image_files[0]}")
    print(f"  Image 2: {image_files[1]}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Extract and match features
    print("\n" + "="*60)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*60)
    
    extractor = FeatureExtractor(method='sift', match_ratio=0.75)
    feature_results = extractor.process_image_pair(img1, img2, visualize=False)
    
    pts1 = feature_results['points1']
    pts2 = feature_results['points2']
    
    # Estimate pose
    print("\n" + "="*60)
    print("STEP 2: POSE ESTIMATION")
    print("="*60)
    
    estimator = PoseEstimator()
    pose_results = estimator.estimate_pose_from_matches(
        pts1, pts2, img1.shape, method='essential'
    )
    
    # Triangulate 3D points
    print("\n" + "="*60)
    print("STEP 3: 3D RECONSTRUCTION")
    print("="*60)
    
    points_3d = estimator.triangulate_points(
        pts1, pts2,
        pose_results['K'],
        pose_results['R'],
        pose_results['t']
    )
    
    # Visualize
    print("\n" + "="*60)
    print("STEP 4: VISUALIZATION")
    print("="*60)
    
    poses = [
        {'R': np.eye(3), 't': np.zeros((3, 1))},  # Camera 1 at origin
        {'R': pose_results['R'], 't': pose_results['t']}  # Camera 2
    ]
    
    os.makedirs("./pose_output", exist_ok=True)
    
    estimator.visualize_camera_poses(
        poses, points_3d,
        save_path="./pose_output/camera_poses.png"
    )
    
    # Save point cloud
    estimator.save_point_cloud(points_3d, "./pose_output/point_cloud.ply")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print("\nResults saved to ./pose_output/")
    print("  - camera_poses.png (visualization)")
    print("  - point_cloud.ply (3D points)")
    print("\nNext steps:")
    print("  1. Open point_cloud.ply in MeshLab or CloudCompare")
    print("  2. Add more images for denser reconstruction")
    print("  3. Next: Build depth estimation and mesh generation")
    print("="*60 + "\n")
    
    return {
        'feature_results': feature_results,
        'pose_results': pose_results,
        'points_3d': points_3d,
        'poses': poses
    }


if __name__ == "__main__":
    results = demo_pose_estimation()