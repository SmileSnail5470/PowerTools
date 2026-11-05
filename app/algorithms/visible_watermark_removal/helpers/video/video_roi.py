import cv2
import shutil
from pathlib import Path
from typing import TypeAlias


BBox: TypeAlias = tuple[int, int, int, int]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def expand_bbox_keep_center(xmin: int, ymin: int, xmax: int, ymax: int, img_w: int, img_h: int, pad: int = 100) -> BBox:
    width, height = xmax - xmin, ymax - ymin
    expanded_width = min(img_w, width + 2 * pad)
    expanded_height = min(img_h, height + 2 * pad)
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    left = max(0, min(img_w - expanded_width, int(center_x - expanded_width / 2)))
    top = max(0, min(img_h - expanded_height, int(center_y - expanded_height / 2)))
    return left, top, left + expanded_width, top + expanded_height


def get_roi(masks_dir: Path, pad: int = 100) -> BBox | None:
    mask_files = _image_files(masks_dir)
    if not mask_files:
        return None
    global_xmin, global_ymin = float("inf"), float("inf")
    global_xmax, global_ymax = float("-inf"), float("-inf")
    reference_shape: tuple[int, int] | None = None
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask image: {mask_path}")
        if reference_shape is None:
            reference_shape = mask.shape
        elif mask.shape != reference_shape:
            raise ValueError(f"Mask dimensions do not match: {mask_path}")
        coords = cv2.findNonZero(mask)
        if coords is None:
            continue
        coords = coords.reshape(-1, 2)
        x, y = coords[:, 0], coords[:, 1]
        global_xmin = min(global_xmin, int(x.min()))
        global_ymin = min(global_ymin, int(y.min()))
        global_xmax = max(global_xmax, int(x.max()))
        global_ymax = max(global_ymax, int(y.max()))
    if global_xmin == float("inf"):
        return None
    assert reference_shape is not None
    height, width = reference_shape
    return expand_bbox_keep_center(global_xmin, global_ymin, global_xmax + 1, global_ymax + 1, width, height, pad)


def crop_frames_and_masks(input_frames_dir: Path, masks_dir: Path, bbox: BBox | None) -> tuple[Path, Path]:
    if bbox is None:
        return input_frames_dir, masks_dir
    cropped_frames_dir = input_frames_dir.parent / f"{input_frames_dir.name}_cropped"
    cropped_masks_dir = masks_dir.parent / f"{masks_dir.name}_cropped"
    cropped_frames_dir.mkdir(parents=True, exist_ok=True)
    cropped_masks_dir.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bbox
    for path in _image_files(input_frames_dir):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read frame image: {path}")
        height, width = image.shape[:2]
        cropped = image[max(0, ymin):min(height, ymax), max(0, xmin):min(width, xmax)]
        if cropped.size == 0:
            raise ValueError(f"ROI does not overlap frame: {path}")
        if not cv2.imwrite(str(cropped_frames_dir / path.name), cropped):
            raise OSError(f"Failed to write cropped frame: {cropped_frames_dir / path.name}")
    for path in _image_files(masks_dir):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask image: {path}")
        height, width = mask.shape[:2]
        cropped = mask[max(0, ymin):min(height, ymax), max(0, xmin):min(width, xmax)]
        if cropped.size == 0:
            raise ValueError(f"ROI does not overlap mask: {path}")
        if not cv2.imwrite(str(cropped_masks_dir / path.name), cropped):
            raise OSError(f"Failed to write cropped mask: {cropped_masks_dir / path.name}")
    return cropped_frames_dir, cropped_masks_dir


def paste_cropped_frames(input_frames_dir: Path, cropped_out_dir: Path, output_dir: Path, bbox: BBox | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cropped_path in _image_files(cropped_out_dir):
        if bbox is None:
            shutil.copy2(cropped_path, output_dir / cropped_path.name)
            continue
        original_path = input_frames_dir / cropped_path.name
        original = cv2.imread(str(original_path))
        cropped = cv2.imread(str(cropped_path))
        if original is None:
            raise ValueError(f"Failed to read frame image: {original_path}")
        if cropped is None:
            raise ValueError(f"Failed to read frame image: {cropped_path}")
        xmin, ymin, xmax, ymax = bbox
        y1, y2 = max(0, ymin), min(original.shape[0], ymax)
        x1, x2 = max(0, xmin), min(original.shape[1], xmax)
        if (y2 - y1, x2 - x1) != cropped.shape[:2]:
            raise ValueError(f"Cropped frame dimensions do not match ROI: {cropped_path}")
        original[y1:y2, x1:x2] = cropped
        if not cv2.imwrite(str(output_dir / cropped_path.name), original):
            raise OSError(f"Failed to write frame image: {output_dir / cropped_path.name}")
