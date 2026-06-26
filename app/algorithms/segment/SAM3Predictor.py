import math
import os
import platform
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from app.algorithms.segment.SimpleTokenizer import SimpleTokenizer
from app.algorithms.segment.preprocessing import preprocess_opencv
ort.preload_dlls(directory="")
from app.algorithms import general_inference_session


@dataclass
class InferenceResult:
    """Inference result (equivalent to SAM3Predictor::InferenceResult)."""
    mask: np.ndarray       # binary mask (H, W) uint8
    score: float           # confidence score
    box: Tuple[float, float, float, float]  # (x, y, w, h) as rectangular box
    label: str             # class label


@dataclass
class InteractiveResult:
    """Internal interactive pipeline result."""
    mask: np.ndarray       # binary mask (H, W) uint8
    score: float           # IoU score



class SAM3Predictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._g_encoder_session: Optional[ort.InferenceSession] = None
        self._lang_session: Optional[ort.InferenceSession] = None
        self._g_decoder_session: Optional[ort.InferenceSession] = None
        self._i_encoder_session: Optional[ort.InferenceSession] = None
        self._i_decoder_session: Optional[ort.InferenceSession] = None
        self.tokenizer = SimpleTokenizer(f"{model_dir}/vocab.txt", f"{model_dir}/merges.txt")

    def _get_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return opts

    def _ensure_grounding_models(self):
        if self._g_encoder_session is not None:
            return
        opts = self._get_session_options()
        providers, provider_options = self._get_providers()

        self._g_encoder_session = general_inference_session(
            os.path.join(self.model_dir, "sam3_grounding_encoder.encmodel"),
            opts,
            providers=providers,
            provider_options=provider_options,
        )
        self._lang_session = general_inference_session(
            os.path.join(self.model_dir, "sam3_language_encoder.encmodel"),
            opts,
            providers=providers,
            provider_options=provider_options,
        )
        self._g_decoder_session = general_inference_session(
            os.path.join(self.model_dir, "sam3_grounding_decoder.encmodel"),
            opts,
            providers=providers,
            provider_options=provider_options,
        )

    def _ensure_interactive_models(self):
        if self._i_encoder_session is not None:
            return
        opts = self._get_session_options()
        providers, provider_options = self._get_providers()

        self._i_encoder_session = general_inference_session(
            os.path.join(self.model_dir, "sam3_encoder.encmodel"),
            opts,
            providers=providers,
            provider_options=provider_options,
        )
        self._i_decoder_session = general_inference_session(
            os.path.join(self.model_dir, "sam3_decoder.encmodel"),
            opts,
            providers=providers,
            provider_options=provider_options,
        )

    def _get_providers(self) -> Tuple[List[str], List[dict]]:
        available = ort.get_available_providers()
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        elif "CUDAExecutionProvider" in available and self._hash_cuda_gpu():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"arena_extend_strategy": "kSameAsRequested"}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        return providers, provider_options
    
    def _hash_cuda_gpu(self):
        if platform.system() != "Windows":
            return True
        cuda_path = r"C:\Program Files\NVIDIA Corporation"
        if os.path.exists(cuda_path):
            return True
        return False

    def predict_text(
        self,
        bgr_img: np.ndarray,
        text: str,
        threshold: float = 0.5,
        max_detections: int = 0,
    ) -> List[InferenceResult]:
        """Text-prompted grounding segmentation.

        Args:
            bgr_img: BGR uint8 image (H, W, 3)
            text: Text prompt (e.g. "person", "cat")
            threshold: Confidence threshold
            max_detections: Max detections (0 = unlimited)

        Returns:
            List of InferenceResult
        """
        self._ensure_grounding_models()
        results = self._run_grounding_inference(bgr_img, text, [], [], threshold, max_detections)
        return results

    def predict_point(
        self,
        bgr_img: np.ndarray,
        point: Tuple[float, float],
        threshold: float = 0.5,
        max_detections: int = 0,
        label: str = "object",
    ) -> List[InferenceResult]:
        """Point-prompted interactive segmentation.

        Args:
            bgr_img: BGR uint8 image (H, W, 3)
            point: (x, y) point coordinates
            threshold: Confidence threshold
            max_detections: Max detections (0 = unlimited)
            label: Class label string

        Returns:
            List of InferenceResult
        """
        self._ensure_interactive_models()
        points = [(point[0], point[1])]
        labels = [1]  # 1 = foreground
        res = self._run_interactive_inference(bgr_img, points, labels, [])

        results: List[InferenceResult] = []
        if res.score > threshold:
            contours, _ = cv2.findContours(res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
            else:
                x = y = w = h = 0.0
            results.append(InferenceResult(
                mask=res.mask,
                score=res.score,
                box=(float(x), float(y), float(w), float(h)),
                label=label,
            ))
        return results

    def predict_box(
        self,
        bgr_img: np.ndarray,
        box: Tuple[float, float, float, float],
        threshold: float = 0.5,
        max_detections: int = 0,
        label: str = "object",
    ) -> List[InferenceResult]:
        """Box-prompted interactive segmentation.

        Args:
            bgr_img: BGR uint8 image (H, W, 3)
            box: (x, y, w, h) bounding box
            threshold: Confidence threshold
            max_detections: Max detections (0 = unlimited)
            label: Class label string

        Returns:
            List of InferenceResult
        """
        self._ensure_interactive_models()
        boxes = [box]
        res = self._run_interactive_inference(bgr_img, [], [], boxes)

        results: List[InferenceResult] = []
        if res.score > threshold:
            contours, _ = cv2.findContours(res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
            else:
                x = y = w = h = 0.0
            results.append(InferenceResult(
                mask=res.mask,
                score=res.score,
                box=(float(x), float(y), float(w), float(h)),
                label=label,
            ))
        return results

    def _run_interactive_inference(
        self,
        bgr_img: np.ndarray,
        points: List[Tuple[float, float]],
        labels: List[int],
        boxes: List[Tuple[float, float, float, float]],
    ) -> InteractiveResult:
        """Run the interactive (point/box) pipeline.

        Equivalent to SAM3Predictor::run_interactive_inference.

        Args:
            bgr_img: BGR uint8 image (H, W, 3)
            points: List of (x, y) points
            labels: List of point labels (1=foreground, 0=background)
            boxes: List of (x, y, w, h) boxes

        Returns:
            InteractiveResult with binary mask and IoU score
        """
        h_img, w_img = bgr_img.shape[:2]
        TARGET_SIZE = (1008, 1008)
        input_blob = self._preprocess_image(bgr_img, target_size=TARGET_SIZE)

        encoder_input_name = self._i_encoder_session.get_inputs()[0].name
        encoder_output_names = [o.name for o in self._i_encoder_session.get_outputs()]
        encoder_outputs = self._i_encoder_session.run(encoder_output_names, {encoder_input_name: input_blob})
        pix_feat = encoder_outputs[0]  # (1, C, H, W)
        high_res_0 = encoder_outputs[1]
        high_res_1 = encoder_outputs[2]

        scale_x = 1008.0 / w_img
        scale_y = 1008.0 / h_img

        final_coords: List[float] = []
        final_labels: List[int] = []

        for px, py in points:
            final_coords.append(px * scale_x)
            final_coords.append(py * scale_y)
        for lbl in labels:
            final_labels.append(lbl)
        for x, y, w, h in boxes:
            final_coords.append(x * scale_x)
            final_coords.append(y * scale_y)
            final_labels.append(2)  # box start label = 2
            final_coords.append((x + w) * scale_x)
            final_coords.append((y + h) * scale_y)
            final_labels.append(3)  # box end label = 3

        if not final_labels:
            final_coords = [0.0, 0.0]
            final_labels = [-1]

        num_pts = len(final_labels)
        coords_arr = np.array(final_coords, dtype=np.float32).reshape(1, num_pts, 2)
        labels_arr = np.array(final_labels, dtype=np.int32).reshape(1, num_pts)

        decoder_inputs = {
            "pix_feat": pix_feat,
            "high_res_0": high_res_0,
            "high_res_1": high_res_1,
            "point_coords": coords_arr,
            "point_labels": labels_arr,
        }
        decoder_output_names = [o.name for o in self._i_decoder_session.get_outputs()]

        decoder_outputs = self._i_decoder_session.run(decoder_output_names, decoder_inputs)
        masks_data = decoder_outputs[0]  # (1, 3, H, W)
        ious_data = decoder_outputs[1]   # (1, 3)

        # Find best mask (max IoU among 3 candidates)
        ious = ious_data[0]  # shape (3,)
        best_idx = int(np.argmax(ious))
        max_iou = float(ious[best_idx])

        # Get mask dimensions
        mask_logit = masks_data[0, best_idx, :, :]  # (mask_h, mask_w)

        # Resize to original image size
        mask_logit_resized = cv2.resize(np.ascontiguousarray(mask_logit, dtype=np.float32), (w_img, h_img))

        # Threshold at 0.0 (sigmoind logit)
        binary_mask = np.where(mask_logit_resized > 0.0, 255, 0).astype(np.uint8)
        return InteractiveResult(mask=binary_mask, score=max_iou)

    def _run_grounding_inference(
        self,
        bgr_img: np.ndarray,
        text: str,
        box_coords_in: List[float],
        box_labels_in: List[int],
        threshold: float,
        max_detections: int,
    ) -> List[InferenceResult]:
        """Run the grounding (text-prompted) pipeline.

        Equivalent to SAM3Predictor::run_grounding_inference.
        """
        h_img, w_img = bgr_img.shape[:2]
        TARGET_SIZE = (1008, 1008)
        input_blob = self._preprocess_image(bgr_img, target_size=TARGET_SIZE)
        enc_input_name = self._g_encoder_session.get_inputs()[0].name
        enc_output_names = [o.name for o in self._g_encoder_session.get_outputs()]
        encoder_outputs = self._g_encoder_session.run(enc_output_names, {enc_input_name: input_blob})

        feat0 = encoder_outputs[0]
        feat1 = encoder_outputs[1]
        feat2 = encoder_outputs[2]
        vpe0 = encoder_outputs[3]
        vpe1 = encoder_outputs[4]
        vpe2 = encoder_outputs[5]

        tokenized = self.tokenizer.tokenize([text], 32)
        tokens_data = np.array(tokenized[0], dtype=np.int64).reshape(1, 32)

        lang_input_name = self._lang_session.get_inputs()[0].name
        lang_output_names = [o.name for o in self._lang_session.get_outputs()]
        lang_outputs = self._lang_session.run(lang_output_names, {lang_input_name: tokens_data})

        text_attention_mask = lang_outputs[0]
        text_memory = lang_outputs[1]
        text_embeds = lang_outputs[2]

        if box_coords_in:
            # Normalize box coordinates to [0, 1]
            box_coords_data = np.array([
                box_coords_in[0] / w_img,
                box_coords_in[1] / h_img,
                box_coords_in[2] / w_img,
                box_coords_in[3] / h_img,
            ], dtype=np.float32).reshape(1, 1, 4)
            box_labels_data = np.array(box_labels_in, dtype=np.int64).reshape(1, 1)
            box_masks_data = np.array([0], dtype=np.bool_).reshape(1, 1)
        else:
            box_coords_data = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32).reshape(1, 1, 4)
            box_labels_data = np.array([1], dtype=np.int64).reshape(1, 1)
            box_masks_data = np.array([1], dtype=np.bool_).reshape(1, 1)

        decoder_input_map = {
            "feat0": feat0,
            "feat1": feat1,
            "feat2": feat2,
            "vpe2": vpe2,
            "lang_mask": text_attention_mask,
            "lang_feat": text_memory,
            "box_coords": box_coords_data,
            "box_labels": box_labels_data,
            "box_masks": box_masks_data,
        }
        # Only include vpe0/vpe1 if the model expects them (matching C++ sends vpe2 only)
        dec_output_names = [o.name for o in self._g_decoder_session.get_outputs()]
        decoder_outputs = self._g_decoder_session.run(dec_output_names, decoder_input_map)

        boxes_arr = decoder_outputs[0]  # (num_prompts, num_queries, 4)
        scores_arr = decoder_outputs[1]  # (num_prompts, num_queries)
        masks_arr = decoder_outputs[2]   # (num_prompts, num_queries, H, W)
        presence_arr = decoder_outputs[3]  # (num_prompts,)

        num_prompts = boxes_arr.shape[0]
        num_queries = boxes_arr.shape[1]
        mask_h = masks_arr.shape[2]
        mask_w = masks_arr.shape[3]

        candidates: List[Tuple[float, InferenceResult]] = []

        for i in range(num_prompts):
            presence_score = 1.0 / (1.0 + math.exp(-float(presence_arr[i])))
            for q in range(num_queries):
                logit = float(scores_arr[i, q])
                score = (1.0 / (1.0 + math.exp(-logit))) * presence_score
                if score < threshold:
                    continue

                box = boxes_arr[i, q, :]  # (4,) in [0,1] normalized
                mask_logit = masks_arr[i, q, :, :]  # (mask_h, mask_w)

                # Resize mask to original image size
                mask_resized = cv2.resize(np.ascontiguousarray(mask_logit, dtype=np.float32), (w_img, h_img))
                binary_mask = np.where(mask_resized > 0.0, 255, 0).astype(np.uint8)

                # Denormalize box to pixel coordinates
                x1 = box[0] * w_img
                y1 = box[1] * h_img
                x2 = box[2] * w_img
                y2 = box[3] * h_img

                label_str = "object" if text == "visual" else text

                candidates.append((
                    score,
                    InferenceResult(
                        mask=binary_mask,
                        score=score,
                        box=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                        label=label_str,
                    ),
                ))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if max_detections > 0:
            candidates = candidates[:max_detections]
        results = [c[1] for c in candidates]
        return results

    @staticmethod
    def _preprocess_image(bgr_img: np.ndarray, target_size=(1008, 1008)) -> np.ndarray:
        return preprocess_opencv(bgr_img, target_size=target_size)

    def __del__(self):
        self._g_encoder_session = None
        self._lang_session = None
        self._g_decoder_session = None
        self._i_encoder_session = None
        self._i_decoder_session = None
