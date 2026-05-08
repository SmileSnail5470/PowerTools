import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.algorithms.segment.SimpleTokenizer import SimpleTokenizer
from app.algorithms.segment.preprocessing import preprocess_opencv


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
    def __init__(
        self,
        model_dir: str,
    ):
        """
        Args:
            model_dir: Path to the directory containing ONNX model files and tokenizer files (vocab.txt, merges.txt).
        """
        self.model_dir = model_dir

        # ONNX Runtime environment
        self._env = ort.get_default()
        # We'll create sessions on demand

        # Grounding pipeline models (lazy loaded)
        self._g_encoder_session: Optional[ort.InferenceSession] = None
        self._lang_session: Optional[ort.InferenceSession] = None
        self._g_decoder_session: Optional[ort.InferenceSession] = None

        # Interactive pipeline models (lazy loaded)
        self._i_encoder_session: Optional[ort.InferenceSession] = None
        self._i_decoder_session: Optional[ort.InferenceSession] = None

        # Tokenizer
        self.tokenizer = SimpleTokenizer(
            f"{model_dir}/vocab.txt",
            f"{model_dir}/merges.txt",
        )

        # GPU preprocessing buffers (not applicable in Python; handled by numpy)
        # No CUDA-specific fields needed

    # ------------------------------------------------------------------
    # Session options
    # ------------------------------------------------------------------

    def _get_session_options(self) -> ort.SessionOptions:
        """Create session options (equivalent to get_session_options)."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # auto
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.use_gpu:
            try:
                opts.register_openvino()
            except Exception:
                pass
        return opts

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_grounding_models(self):
        """Load grounding pipeline models if not already loaded."""
        if self._g_encoder_session is not None:
            return
        print("Loading Grounding/Text models...")
        t0 = time.time()

        opts = self._get_session_options()
        providers = self._get_providers()

        self._g_encoder_session = ort.InferenceSession(
            f"{self.model_dir}/sam3_grounding_encoder.onnx",
            opts,
            providers=providers,
        )
        self._lang_session = ort.InferenceSession(
            f"{self.model_dir}/sam3_language_encoder.onnx",
            opts,
            providers=providers,
        )
        self._g_decoder_session = ort.InferenceSession(
            f"{self.model_dir}/sam3_grounding_decoder.onnx",
            opts,
            providers=providers,
        )

        t1 = time.time()
        print(f"Model loading time: {(t1 - t0) * 1000:.0f} ms")

    def _ensure_interactive_models(self):
        """Load interactive pipeline models if not already loaded."""
        if self._i_encoder_session is not None:
            return
        print("Loading Interactive (Point/Box) models...")
        t0 = time.time()

        opts = self._get_session_options()
        providers = self._get_providers()

        self._i_encoder_session = ort.InferenceSession(
            f"{self.model_dir}/sam3_encoder.onnx",
            opts,
            providers=providers,
        )
        self._i_decoder_session = ort.InferenceSession(
            f"{self.model_dir}/sam3_decoder.onnx",
            opts,
            providers=providers,
        )

        t1 = time.time()
        print(f"Model loading time: {(t1 - t0) * 1000:.0f} ms")

    def _get_providers(self) -> List[str]:
        """Get the list of providers based on use_gpu flag."""
        if self.use_gpu:
            try:
                import onnxruntime.capi._pybind_state as _  # noqa: F401
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                print("[Warning] CUDA EP not available, falling back to CPU")
                return ["CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self):
        """Warm up models (eager loading)."""
        print("Warming up SAM3 models...")
        self._ensure_grounding_models()
        self._ensure_interactive_models()
        print("Warmup completed. Models are ready in memory.")

    def predict_text(
        self,
        bgr_img: np.ndarray,
        text: str,
        threshold: float = 0.25,
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

        t0 = time.time()
        results = self._run_grounding_inference(
            bgr_img, text, [], [], threshold, max_detections
        )
        t1 = time.time()
        duration = (t1 - t0) * 1000
        print(f"Pure Inference time (Preprocessing + Forward): {duration:.0f} ms")
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

        t0 = time.time()
        points = [(point[0], point[1])]
        labels = [1]  # 1 = foreground
        res = self._run_interactive_inference(bgr_img, points, labels, [])

        results: List[InferenceResult] = []
        if res.score > threshold:
            # Compute bounding rect from mask (equivalent to cv::boundingRect)
            contours, _ = cv2.findContours(
                res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
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

        t1 = time.time()
        duration = (t1 - t0) * 1000
        print(f"Inference time: {duration:.0f} ms")
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

        t0 = time.time()
        boxes = [box]
        res = self._run_interactive_inference(bgr_img, [], [], boxes)

        results: List[InferenceResult] = []
        if res.score > threshold:
            contours, _ = cv2.findContours(
                res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
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

        t1 = time.time()
        duration = (t1 - t0) * 1000
        print(f"Inference time: {duration:.0f} ms")
        return results

    # ------------------------------------------------------------------
    # Interactive pipeline internals
    # ------------------------------------------------------------------

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

        # 1. Image preprocessing → encoder input
        # Equivalent to the CPU fallback in C++:
        #   cv::cvtColor -> cv::dnn::blobFromImage
        input_blob = self._preprocess_image(bgr_img)

        # Encoder inputs
        encoder_input_name = self._i_encoder_session.get_inputs()[0].name
        encoder_output_names = [o.name for o in self._i_encoder_session.get_outputs()]
        encoder_outputs = self._i_encoder_session.run(
            encoder_output_names,
            {encoder_input_name: input_blob},
        )
        pix_feat = encoder_outputs[0]  # (1, C, H, W)
        high_res_0 = encoder_outputs[1]
        high_res_1 = encoder_outputs[2]

        # 2. Prepare decoder inputs (points + boxes)
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

        # Decoder input names (matching C++ decoder_input_names)
        # pix_feat, high_res_0, high_res_1, point_coords, point_labels
        decoder_inputs = {
            "pix_feat": pix_feat,
            "high_res_0": high_res_0,
            "high_res_1": high_res_1,
            "point_coords": coords_arr,
            "point_labels": labels_arr,
        }
        decoder_output_names = [o.name for o in self._i_decoder_session.get_outputs()]

        decoder_outputs = self._i_decoder_session.run(
            decoder_output_names, decoder_inputs
        )
        masks_data = decoder_outputs[0]  # (1, 3, H, W)
        ious_data = decoder_outputs[1]   # (1, 3)

        # Find best mask (max IoU among 3 candidates)
        ious = ious_data[0]  # shape (3,)
        best_idx = int(np.argmax(ious))
        max_iou = float(ious[best_idx])

        # Get mask dimensions
        mask_h = masks_data.shape[2]
        mask_w = masks_data.shape[3]
        mask_logit = masks_data[0, best_idx, :, :]  # (mask_h, mask_w)

        # Resize to original image size
        mask_logit_resized = cv2.resize(mask_logit, (w_img, h_img))

        # Threshold at 0.0 (sigmoind logit)
        binary_mask = np.where(mask_logit_resized > 0.0, 255, 0).astype(np.uint8)

        return InteractiveResult(mask=binary_mask, score=max_iou)

    # ------------------------------------------------------------------
    # Grounding pipeline internals
    # ------------------------------------------------------------------

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

        # 1. Image preprocessing
        t0 = time.time()
        input_blob = self._preprocess_image(bgr_img)
        t1 = time.time()
        print(f"Preprocessing time: {(t1 - t0) * 1000:.0f} ms")

        # 2. Grounding encoder inference
        try:
            enc_input_name = self._g_encoder_session.get_inputs()[0].name
            enc_output_names = [o.name for o in self._g_encoder_session.get_outputs()]
            t2 = time.time()
            encoder_outputs = self._g_encoder_session.run(
                enc_output_names, {enc_input_name: input_blob}
            )
            t3 = time.time()
            print(f"Image encoder time: {(t3 - t2) * 1000:.0f} ms")
        except Exception as e:
            if self.use_gpu:
                print(f"[Warning] GPU Inference failed (likely OOM: {e}). Falling back to CPU...")
                self.use_gpu = False
                self._g_encoder_session = None
                self._lang_session = None
                self._g_decoder_session = None
                return self._run_grounding_inference(
                    bgr_img, text, box_coords_in, box_labels_in, threshold, max_detections
                )
            raise e

        feat0 = encoder_outputs[0]
        feat1 = encoder_outputs[1]
        feat2 = encoder_outputs[2]
        vpe0 = encoder_outputs[3]
        vpe1 = encoder_outputs[4]
        vpe2 = encoder_outputs[5]

        # 3. Language encoder inference
        tokenized = self.tokenizer.tokenize([text], 32)
        tokens_data = np.array(tokenized[0], dtype=np.int64).reshape(1, 32)

        lang_input_name = self._lang_session.get_inputs()[0].name
        lang_output_names = [o.name for o in self._lang_session.get_outputs()]
        t4 = time.time()
        lang_outputs = self._lang_session.run(
            lang_output_names, {lang_input_name: tokens_data}
        )
        t5 = time.time()
        print(f"Language encoder time: {(t5 - t4) * 1000:.0f} ms")

        text_attention_mask = lang_outputs[0]
        text_memory = lang_outputs[1]
        text_embeds = lang_outputs[2]

        # 4. Grounding decoder inference
        # Prepare box prompt (default: empty prompt with padding)
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

        t6 = time.time()
        decoder_outputs = self._g_decoder_session.run(
            dec_output_names, decoder_input_map
        )
        t7 = time.time()
        print(f"Grounding decoder time: {(t7 - t6) * 1000:.0f} ms")

        boxes_arr = decoder_outputs[0]  # (num_prompts, num_queries, 4)
        scores_arr = decoder_outputs[1]  # (num_prompts, num_queries)
        masks_arr = decoder_outputs[2]   # (num_prompts, num_queries, H, W)
        presence_arr = decoder_outputs[3]  # (num_prompts,)

        # 5. Postprocessing
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
                mask_resized = cv2.resize(mask_logit, (w_img, h_img))
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

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Take top-k
        if max_detections > 0:
            candidates = candidates[:max_detections]

        results = [c[1] for c in candidates]
        return results

    # ------------------------------------------------------------------
    # Helper: image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_image(bgr_img: np.ndarray) -> np.ndarray:
        """Preprocess a BGR image to model input.

        Equivalent to the CPU fallback path in C++:
            cv::cvtColor + cv::dnn::blobFromImage
        """
        return preprocess_opencv(bgr_img, target_size=(1008, 1008))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        """Destructor equivalent — release sessions."""
        self._g_encoder_session = None
        self._lang_session = None
        self._g_decoder_session = None
        self._i_encoder_session = None
        self._i_decoder_session = None
