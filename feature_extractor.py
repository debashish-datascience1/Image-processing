import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os


class FeatureExtractor:
    """
    Extract and match features between images for 3D reconstruction.
    This is the first step in converting 2D panoramas to 3D models.
    """
    
    def __init__(self, method='sift', match_ratio=0.75):
        """
        Initialize feature extractor.
        
        Args:
            method: 'sift', 'orb', or 'akaze'
            match_ratio: Lowe's ratio test threshold (0.7-0.8 recommended)
        """
        self.method = method.lower()
        self.match_ratio = match_ratio
        self.detector = self._create_detector()
        self.matcher = cv2.BFMatcher()
        
    def _create_detector(self):
        """Create feature detector based on method"""
        if self.method == 'sift':
            return cv2.SIFT_create(nfeatures=5000)
        elif self.method == 'orb':
            return cv2.ORB_create(nfeatures=5000)
        elif self.method == 'akaze':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract keypoints and descriptors from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: numpy array of feature descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        print(f"  Found {len(keypoints)} keypoints using {self.method.upper()}")
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match features between two sets of descriptors using ratio test.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            good_matches: List of good cv2.DMatch objects
        """
        # Handle ORB (binary descriptors) vs SIFT (float descriptors)
        if self.method == 'orb' or self.method == 'akaze':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Find 2 best matches for each descriptor (for ratio test)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        print(f"  Found {len(good_matches)} good matches (ratio test: {self.match_ratio})")
        
        return good_matches
    
    def filter_matches_ransac(self, kp1: List, kp2: List, matches: List,
                              ransac_threshold=5.0) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Filter matches using RANSAC to remove outliers.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of DMatch objects
            ransac_threshold: RANSAC reprojection threshold in pixels
            
        Returns:
            inlier_matches: Filtered matches
            fundamental_matrix: Estimated fundamental matrix
            mask: Inlier mask
        """
        if len(matches) < 8:
            print("  Warning: Too few matches for RANSAC")
            return matches, None, None
        
        # Extract matched point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Estimate fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_threshold)
        
        # Filter inlier matches
        if mask is not None:
            inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
            print(f"  RANSAC: {len(inlier_matches)}/{len(matches)} inliers")
        else:
            inlier_matches = matches
            print("  RANSAC failed, keeping all matches")
        
        return inlier_matches, F, mask
    
    def get_matched_points(self, kp1: List, kp2: List, matches: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract coordinates of matched keypoints.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of DMatch objects
            
        Returns:
            pts1: Nx2 array of points from first image
            pts2: Nx2 array of points from second image
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        return pts1, pts2
    
    def visualize_keypoints(self, image: np.ndarray, keypoints: List, 
                           title: str = "Keypoints", save_path: str = None):
        """
        Visualize detected keypoints on an image.
        
        Args:
            image: Input image
            keypoints: List of keypoints
            title: Plot title
            save_path: Optional path to save the image
        """
        # Draw keypoints
        img_kp = cv2.drawKeypoints(image, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                   color=(0, 255, 0))
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} ({len(keypoints)} keypoints)")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved keypoints visualization: {save_path}")
        
        plt.show()
    
    def visualize_matches(self, img1: np.ndarray, kp1: List, 
                         img2: np.ndarray, kp2: List, 
                         matches: List, title: str = "Feature Matches",
                         save_path: str = None, max_matches: int = 100):
        """
        Visualize feature matches between two images.
        
        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: List of matches
            title: Plot title
            save_path: Optional path to save
            max_matches: Maximum number of matches to draw
        """
        # Limit matches for cleaner visualization
        matches_to_draw = matches[:max_matches] if len(matches) > max_matches else matches
        
        # Draw matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_to_draw, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                     matchColor=(0, 255, 0),
                                     singlePointColor=(255, 0, 0))
        
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} ({len(matches)} total, showing {len(matches_to_draw)})")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved matches visualization: {save_path}")
        
        plt.show()
    
    def process_image_pair(self, img1: np.ndarray, img2: np.ndarray,
                          visualize: bool = True, save_dir: str = None) -> Dict:
        """
        Complete feature extraction and matching pipeline for an image pair.
        
        Args:
            img1: First image
            img2: Second image
            visualize: Whether to show visualizations
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary containing keypoints, descriptors, matches, and points
        """
        print("\n" + "="*60)
        print("FEATURE EXTRACTION & MATCHING")
        print("="*60)
        
        # Extract features from both images
        print("\n[1/4] Extracting features from Image 1...")
        kp1, desc1 = self.extract_features(img1)
        
        print("\n[2/4] Extracting features from Image 2...")
        kp2, desc2 = self.extract_features(img2)
        
        # Match features
        print("\n[3/4] Matching features...")
        matches = self.match_features(desc1, desc2)
        
        # Filter with RANSAC
        print("\n[4/4] Filtering matches with RANSAC...")
        inlier_matches, F, mask = self.filter_matches_ransac(kp1, kp2, matches)
        
        # Get matched point coordinates
        pts1, pts2 = self.get_matched_points(kp1, kp2, inlier_matches)
        
        print("\n" + "="*60)
        print(f"SUMMARY: {len(inlier_matches)} reliable matches found")
        print("="*60 + "\n")
        
        # Visualizations
        if visualize:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            # Visualize keypoints
            self.visualize_keypoints(img1, kp1, "Image 1 - Keypoints",
                                   f"{save_dir}/keypoints_1.png" if save_dir else None)
            self.visualize_keypoints(img2, kp2, "Image 2 - Keypoints",
                                   f"{save_dir}/keypoints_2.png" if save_dir else None)
            
            # Visualize matches
            self.visualize_matches(img1, kp1, img2, kp2, inlier_matches,
                                 "Feature Matches (After RANSAC)",
                                 f"{save_dir}/matches.png" if save_dir else None)
        
        return {
            'keypoints1': kp1,
            'keypoints2': kp2,
            'descriptors1': desc1,
            'descriptors2': desc2,
            'matches': inlier_matches,
            'points1': pts1,
            'points2': pts2,
            'fundamental_matrix': F,
            'mask': mask
        }


def demo_feature_extraction():
    """
    Demo: Extract and match features from two images.
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTOR DEMO")
    print("="*60)
    
    # Check if demo images exist
    img_dir = "./images"
    if not os.path.exists(img_dir):
        print(f"\nError: {img_dir} directory not found!")
        print("Please create './images' folder and add at least 2 images.")
        return
    
    # Get image files
    image_files = sorted([f for f in os.listdir(img_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if len(image_files) < 2:
        print(f"\nError: Need at least 2 images in {img_dir}")
        print(f"Found only: {image_files}")
        return
    
    # Load first two images
    img1_path = os.path.join(img_dir, image_files[0])
    img2_path = os.path.join(img_dir, image_files[1])
    
    print(f"\nLoading images:")
    print(f"  Image 1: {image_files[0]}")
    print(f"  Image 2: {image_files[1]}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("\nError: Failed to load images!")
        return
    
    print(f"\nImage 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    # Create feature extractor
    extractor = FeatureExtractor(method='sift', match_ratio=0.75)
    
    # Process image pair
    results = extractor.process_image_pair(
        img1, img2,
        visualize=True,
        save_dir="./feature_output"
    )
    
    # Print statistics
    print("\nFeature Extraction Results:")
    print(f"  Keypoints in Image 1: {len(results['keypoints1'])}")
    print(f"  Keypoints in Image 2: {len(results['keypoints2'])}")
    print(f"  Good matches: {len(results['matches'])}")
    print(f"  Matched points shape: {results['points1'].shape}")
    
    if results['fundamental_matrix'] is not None:
        print(f"\nFundamental Matrix:")
        print(results['fundamental_matrix'])
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("✓ Features extracted successfully!")
    print("→ Next: Use these matched points for camera pose estimation")
    print("→ Then: Triangulate 3D points from 2D matches")
    print("→ Finally: Build 3D point cloud and mesh")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    # Run demo
    results = demo_feature_extraction()
    
    # Optional: Compare different feature detectors
    if results is not None:
        print("\nWant to try different feature detectors?")
        print("Uncomment the code below to compare SIFT, ORB, and AKAZE:\n")
        
        # Uncomment to compare methods:
        # for method in ['sift', 'orb', 'akaze']:
        #     print(f"\n{'='*60}")
        #     print(f"Testing {method.upper()}")
        #     print(f"{'='*60}")
        #     extractor = FeatureExtractor(method=method, match_ratio=0.75)
        #     
        #     img1 = cv2.imread("./images/image_0.jpg")
        #     img2 = cv2.imread("./images/image_1.jpg")
        #     
        #     results = extractor.process_image_pair(img1, img2, visualize=False)
        #     print(f"{method.upper()}: {len(results['matches'])} matches")