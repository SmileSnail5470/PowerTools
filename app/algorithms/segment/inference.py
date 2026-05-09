from typing import List, Tuple
import cv2
import numpy as np
from app.algorithms.segment.SAM3Predictor import SAM3Predictor, InferenceResult


PALETTE = [
    (255, 255, 0)
]


def get_color(index: int) -> Tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]

def save_visualization(
        img: np.ndarray,
        results: List[InferenceResult],
        filename: str,
        prompt_points: List[Tuple[float, float]] = None,
        prompt_boxes: List[Tuple[float, float, float, float]] = None,
    ):
    prompt_points = [] if prompt_points is None else prompt_points
    prompt_boxes = [] if prompt_boxes is None else prompt_boxes
    overlay = img.copy()
    for i, r in enumerate(results):
        color = get_color(i)
        b, g, r_ = color
        mask = r.mask > 0
        overlay[mask] = (
            overlay[mask] * 0.45
            + np.array([b, g, r_]) * 0.55
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            r.mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    for px, py in prompt_points:
        pt = (int(px), int(py))
        # Outer black circle
        cv2.circle(overlay, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)
        # Inner white dot
        cv2.circle(overlay, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
    for bx, by, bw, bh in prompt_boxes:
        cv2.rectangle(
            overlay,
            (int(bx), int(by)),
            (int(bx + bw), int(by + bh)),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(filename, overlay)


class SegmentationInference():
    def __init__(
            self, 
            model_dir: str, 
            prompt_mode: str = "texts",       # "texts", "points", or "boxes"
            prompt_value: str = "watermark",  # Text prompt or "x,y" for points or "x1,y1,x2,y2" for boxes
            threshold: float = 0.25,          # Confidence threshold
            max_detections: int = 0,          # Max detections (0 = unlimited)
            label: str = "object"             # Class label for points/boxes   
        ):
        self.model_dir = model_dir
        self.prompt_mode = prompt_mode
        self.prompt_value = prompt_value
        self.threshold = threshold
        self.max_detections = max_detections
        self.label = label

    def _save_mask(self, img, results, output_mask_path):
        h, w = img.shape[:2]
        mask_image = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            mask_image[r.mask > 0] = 255
        if output_mask_path is not None:
            cv2.imwrite(output_mask_path, mask_image)
        return mask_image

    def prepare(self):
        self.predictor = SAM3Predictor(self.model_dir)

    def inference_image(self, input_image_path: str, output_mask_path: str | None = None, output_visualization_path: str | None = None) -> np.ndarray:
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError(f"Could not read image: {input_image_path}")
        results: List[InferenceResult] = []
        prompt_points: List[Tuple[float, float]] = []
        prompt_boxes: List[Tuple[float, float, float, float]] = []

        if self.prompt_mode == "texts":
            class_names = [c.strip() for c in self.prompt_value.split(",") if c.strip()]
            for cls in class_names:
                cls_results = self.predictor.predict_text(img, cls, threshold=self.threshold, max_detections=self.max_detections)
                results.extend(cls_results)
        elif self.prompt_mode == "points":
            parts = self.prompt_value.split(",")
            if len(parts) < 2:
                raise ValueError("Error: points mode expects 'x,y' format")
            x, y = float(parts[0].strip()), float(parts[1].strip())
            prompt_points.append((x, y))
            results = self.predictor.predict_point(img, (x, y), threshold=self.threshold, max_detections=self.max_detections, label=self.label)
        elif self.prompt_mode == "boxes":
            parts = self.prompt_value.split(",")
            if len(parts) < 4:
                raise ValueError("Error: boxes mode expects 'x1,y1,x2,y2' format")
            x1, y1 = float(parts[0].strip()), float(parts[1].strip())
            x2, y2 = float(parts[2].strip()), float(parts[3].strip())
            bw, bh = x2 - x1, y2 - y1
            prompt_boxes.append((x1, y1, bw, bh))
            results = self.predictor.predict_box(img, (x1, y1, bw, bh), threshold=self.threshold, max_detections=self.max_detections, label=self.label)
        if output_visualization_path is not None:
            save_visualization(img, results, output_visualization_path, prompt_points, prompt_boxes)
        mask_image = self._save_mask(img, results, output_mask_path)
        return mask_image # [H, W] 0-255 uint8


if __name__ == "__main__":
    import os
    import time
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "deps", "segment")
    input_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "test2.jpg")
    output_mask_path = "output_mask.png"
    output_visualization_path = "output_visualization.png"
    inference = SegmentationInference(
        model_dir=model_dir,
        prompt_mode="texts",
        prompt_value="watermarks",
        threshold=0.25,
        max_detections=0,
        label="watermark"
    )
    inference.prepare()
    start_time = time.time()
    inference.inference_image(input_image_path, output_mask_path, output_visualization_path)
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds. Output saved to {output_mask_path}")