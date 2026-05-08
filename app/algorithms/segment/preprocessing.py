"""
CUDA kernel equivalent for image preprocessing:
  BGR/RGB uint8 → CHW RGB float32 with normalization (val - 127.5) / 127.5
"""

import numpy as np
import cv2


def preprocess_cpu(
    image: np.ndarray, is_bgr: bool = True
) -> np.ndarray:
    """
    CPU preprocessing: replicates the CUDA kernel logic.

    Args:
        image: HWC uint8 image (height, width, 3)
        is_bgr: True if input is BGR (OpenCV default), False if RGB

    Returns:
        CHW float32 array (1, 3, H, W) normalized by (val - 127.5) / 127.5
    """
    h, w = image.shape[:2]

    if is_bgr:
        # BGR → RGB conversion inline (matching CUDA kernel)
        b = (image[:, :, 0].astype(np.float32) - 127.5) / 127.5
        g = (image[:, :, 1].astype(np.float32) - 127.5) / 127.5
        r = (image[:, :, 2].astype(np.float32) - 127.5) / 127.5
    else:
        r = (image[:, :, 0].astype(np.float32) - 127.5) / 127.5
        g = (image[:, :, 1].astype(np.float32) - 127.5) / 127.5
        b = (image[:, :, 2].astype(np.float32) - 127.5) / 127.5

    # CHW layout
    output = np.stack([r, g, b], axis=0).astype(np.float32)
    # Add batch dimension -> (1, 3, H, W)
    output = output[np.newaxis, :, :, :]
    return output


def preprocess_opencv(
    image: np.ndarray, target_size: tuple = (1008, 1008)
) -> np.ndarray:
    """
    CPU preprocessing using OpenCV (equivalent to the CPU fallback path in C++).

    Replicates:
        cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
        cv::dnn::blobFromImage(rgb_img, 1.0/127.5, Size(1008,1008),
                               Scalar(127.5,127.5,127.5), true, false, CV_32F);

    Args:
        image: HWC BGR uint8 image
        target_size: (width, height) for resize

    Returns:
        CHW float32 blob (1, 3, H, W)
    """
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(
        rgb_img,
        scalefactor=1.0 / 127.5,
        size=target_size,
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
        ddepth=cv2.CV_32F,
    )
    return blob
