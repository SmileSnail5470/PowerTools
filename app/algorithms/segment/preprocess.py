import numpy as np
import cv2


def preprocess_bgr(image_bgr: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
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
    target_size = (dst_width, dst_height)
    print(target_size)
    rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
