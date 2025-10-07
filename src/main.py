import os
import sys

# Ensure project root is on sys.path so sibling modules are importable when running from src/
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DDetectionSubmodule.find_defects import CoinDefectDetector
from CDetectionSubmodule.coins_contour_detection import counting_contours
import numpy as np
from scipy import ndimage
from typing import Tuple, Any, List
from dataclasses import dataclass
import cv2


@dataclass
class Defect:
    img_i: int
    ssim: np.floating[Any]
    contours: List[np.ndarray]
    considered_defect: bool


def calculate_ssim(
    image1: cv2.typing.MatLike,
    image2: cv2.typing.MatLike,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    L: float = 255.0,
) -> Tuple[np.floating[Any], cv2.typing.MatLike]:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        image1 (np.ndarray): First image (reference)
        image2 (np.ndarray): Second image (comparison)
        window_size (int): Size of the sliding window (must be odd)
        k1 (float): Constant for luminance comparison
        k2 (float): Constant for contrast comparison
        L (float): Dynamic range of pixel values

    Returns:
        Tuple[float, np.ndarray]: SSIM score and SSIM map
    """

    # Ensure images are the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Convert to float if needed
    if image1.dtype != np.float64:
        image1 = image1.astype(np.float64)
    if image2.dtype != np.float64:
        image2 = image2.astype(np.float64)

    # Constants
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Create Gaussian window
    window = _create_gaussian_window(window_size)

    # Calculate means
    mu1 = ndimage.convolve(image1, window, mode="constant")
    mu2 = ndimage.convolve(image2, window, mode="constant")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = ndimage.convolve(image1 * image1, window, mode="constant") - mu1_sq
    sigma2_sq = ndimage.convolve(image2 * image2, window, mode="constant") - mu2_sq
    sigma12 = ndimage.convolve(image1 * image2, window, mode="constant") - mu1_mu2

    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    ssim_score = np.mean(ssim_map)

    return ssim_score, ssim_map


def _create_gaussian_window(window_size: int, sigma: float = 1.5) -> np.ndarray:
    """Create a Gaussian window for SSIM calculation."""
    window = np.zeros((window_size, window_size))
    center = window_size // 2

    for i in range(window_size):
        for j in range(window_size):
            window[i, j] = np.exp(
                -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma**2)
            )

    return window / np.sum(window)


def find_defects(
    image_paths: List[str],
    reference_index: int,
    ssim_threshold: float,
    include_non_defects: bool = False,
) -> List[Defect]:
    """
    Find defects in coin images.
    """
    defect_detector = CoinDefectDetector()
    defects = []
    reference_image = cv2.imread(image_paths[reference_index], cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        raise ValueError(f"Reference image {image_paths[reference_index]} not found")
    for i, image_path in enumerate(image_paths):
        if i == reference_index:
            continue
        defect_dict = defect_detector.detect_defects(
            image_paths=[image_path, image_paths[reference_index]], reference_index=1
        )
        if not defect_dict["success"]:
            print(f"Defect not found in image {image_path}")
            continue
        defect_contours = defect_dict["defect_contours"]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image {image_path} not found")
        bboxes = []
        for contour in defect_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
        for bbox in bboxes:
            x, y, w, h = bbox
            defect_image = image[y : y + h, x : x + w]
            cut_reference_image = reference_image[y : y + h, x : x + w]
            ssim, ssim_map = calculate_ssim(cut_reference_image, defect_image)
            if ssim < ssim_threshold:
                defects.append(
                    Defect(
                        img_i=i,
                        ssim=ssim,
                        contours=defect_contours,
                        considered_defect=True,
                    )
                )
            elif include_non_defects:
                defects.append(
                    Defect(
                        img_i=i,
                        ssim=ssim,
                        contours=defect_contours,
                        considered_defect=False,
                    )
                )
    return defects


def visualize_defects(
    defects: List[Defect], image_paths: List[str], reference_index: int
):
    reference_image = cv2.imread(image_paths[reference_index])
    if reference_image is None:
        raise ValueError(f"Reference image {image_paths[reference_index]} not found")
    cv2.imshow("Reference", reference_image)
    for defect in defects:
        image = cv2.imread(image_paths[defect.img_i])
        if image is None:
            raise ValueError(f"Image {image_paths[defect.img_i]} not found")
        for contour in defect.contours:
            cv2.drawContours(
                image,
                [contour],
                -1,
                (0, 0, 255) if defect.considered_defect else (0, 255, 0),
                3,
            )
        cv2.imshow("Defect", image)
        q = cv2.waitKey(0)
        if q == 27:
            break
    cv2.destroyWindow("Reference")
    cv2.destroyWindow("Defect")


if __name__ == "__main__":
    import argparse
    import os

    # Change --images to --image and accept a directory path
    parser = argparse.ArgumentParser(description="Find defects in coin images")
    parser.add_argument(
        "-i", "--image", required=True, help="Path to directory containing coin images"
    )
    parser.add_argument(
        "-r", "--reference", type=int, required=True, help="Index of reference image"
    )
    parser.add_argument(
        "-s", "--ssim", type=float, required=True, help="SSIM threshold"
    )
    parser.add_argument(
        "-n", "--non-defects", type=bool, required=True, help="Include non-defects"
    )
    args = parser.parse_args()

    # Collect all image files from the given directory
    image_dir = args.image
    if not os.path.isdir(image_dir):
        raise ValueError(f"Directory {image_dir} does not exist")
    image_paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_paths:
        raise ValueError(f"No image files found in directory {image_dir}")

    reference_index = args.reference
    ssim_threshold = args.ssim
    include_non_defects = args.non_defects

    defects = find_defects(
        image_paths, reference_index, ssim_threshold, include_non_defects
    )
    visualize_defects(defects, image_paths, reference_index)
