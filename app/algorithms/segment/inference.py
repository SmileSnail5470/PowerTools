from typing import List, Tuple
import cv2
import numpy as np
from app.algorithms.segment.SAM3Predictor import SAM3Predictor, InferenceResult


PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (128, 0, 255),
    (0, 128, 255),
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
        colored_mask = np.zeros_like(img)
        colored_mask[r.mask > 0] = [b, g, r_]
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        x, y, w, h = map(int, r.box)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (b, g, r_), 2)

        text = f"{r.label} {r.score:.4f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_w, text_h, baseline = cv2.getTextSize(text, font, font_scale, thickness)

        label_x = max(0, x)
        label_y = max(text_h + 5, y)
        label_w = min(text_w, overlay.shape[1] - label_x)
        label_h = text_h + 5

        if label_w > 0 and label_h > 0 and label_x + label_w <= overlay.shape[1] and label_y >= label_h:
            label_bg = overlay[label_y - label_h : label_y, label_x : label_x + label_w].copy()
            cv2.rectangle(
                label_bg,
                (0, 0),
                (label_w, label_h),
                (b, g, r_),
                -1,
            )
            overlay[label_y - label_h : label_y, label_x : label_x + label_w] = cv2.addWeighted(
                overlay[label_y - label_h : label_y, label_x : label_x + label_w],
                0.6,
                label_bg,
                0.4,
                0,
            )
            cv2.putText(
                overlay,
                text,
                (label_x, label_y - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
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

    def prepare(self):
        self.predictor = SAM3Predictor(self.model_dir)

    def inference_image(self, input_image_path: str, output_image_path: str):
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
        save_visualization(img, results, output_image_path, prompt_points, prompt_boxes)