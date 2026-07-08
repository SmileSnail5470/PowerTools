import os
import cv2
import numpy as np
from app.algorithms.segment.sam3_pcs import SAM3_PCS
from app.algorithms.segment.tokenizer import load_tokenizer


CONTEXT_LENGTH = 32


class SegmentationInference():
    def __init__(
            self, 
            model_dir: str, 
            prompt_mode: str = "texts",       # "texts", "points", or "boxes"
            prompt_value: str = "watermark",  # Text prompt or ["x,y"] for points or ["x1,y1,x2,y2"] for boxes
            threshold: float = 0.5,           # Confidence threshold 
        ):
        self.model_dir = model_dir
        self.prompt_mode = prompt_mode
        self.prompt_value = prompt_value
        self.threshold = threshold

    def _save_mask(self, img, results):
        h, w = img.shape[:2]
        mask_image = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            mask_image[r > 0] = 255
        return mask_image

    def tokenize_prompt(self, prompt: str):
        tokenizer_json = os.path.join(self.model_dir, "tokenizer.json")
        bpe_path = os.path.join(self.model_dir, "bpe_simple_vocab_16e6.txt.gz")
        tok = load_tokenizer(context_length=CONTEXT_LENGTH, tokenizer_json=tokenizer_json, bpe_path=bpe_path)
        input_ids, attention_mask = tok.tokenize(prompt)
        return input_ids, attention_mask
    
    def prepare(self):
        model_path = os.path.join(self.model_dir, "sam3.encmodel")
        self.pcs = SAM3_PCS(model_path, self.threshold)

    def inference_image(self, input_image_path: str) -> np.ndarray:
        img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_image_path}")
        if self.prompt_mode == "texts":
            results = []
            class_names = [c.strip() for c in self.prompt_value.split(",") if c.strip()]
            for cls in class_names:
                input_ids, attention_mask = self.tokenize_prompt(cls)
                self.pcs.set_prompt(input_ids, attention_mask)
                mask = self.pcs.infer_on_image(img)
                results.append(mask)
            mask_image = self._save_mask(img, results)
        elif self.prompt_mode == "boxes":
            input_ids, attention_mask = self.tokenize_prompt("watermark")
            self.pcs.set_prompt(input_ids, attention_mask)
            mask = self.pcs.infer_on_image(img)
            h, w = img.shape[:2]
            mask_image = np.zeros((h, w), dtype=np.uint8)
            for one_prompt_value in self.prompt_value:
                parts = one_prompt_value.split(",")
                if len(parts) < 4:
                    raise ValueError("Error: boxes mode expects 'x1,y1,x2,y2' format")
                x1, y1 = int(parts[0].strip()), int(parts[1].strip())
                x2, y2 = int(parts[2].strip()), int(parts[3].strip())
                mask_image[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return mask_image