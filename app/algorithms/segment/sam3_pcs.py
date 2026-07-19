from typing import List, Optional
import numpy as np
import onnxruntime as ort
import app.algorithms.segment.postprocess as pp
from app.algorithms.segment.preprocess import preprocess_bgr
from app.algorithms import general_provider, general_session, general_inference_session, ORTEnvironment
ORTEnvironment.initialize()


class SAM3_PCS:
    def __init__(self, model_path: str, prob_threshold: float = 0.5):
        self._probability_threshold = prob_threshold

        providers, provider_options = general_provider()
        self.session = general_inference_session(
            model_path=model_path,
            sess_options=self._get_session_options(),
            providers=providers,
            provider_options=provider_options
        )

        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

        self._pixel_input = self._find_input("pixel_values")
        pv = next(i for i in self.session.get_inputs() if i.name == self._pixel_input)
        # Shape is [1, 3, H, W]; fall back to 1008 if a dim is dynamic.
        self.in_height = pv.shape[2] if isinstance(pv.shape[2], int) else 1008
        self.in_width = pv.shape[3] if isinstance(pv.shape[3], int) else 1008

        self._input_ids_name = self._find_input("input_ids")
        self._attention_mask_name = self._find_input("attention_mask")

        self._input_ids: Optional[np.ndarray] = None
        self._attention_mask: Optional[np.ndarray] = None


    def _get_session_options(self) -> ort.SessionOptions:
        opts = general_session()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return opts

    def _find_input(self, name: str) -> str:
        if name in self._input_names:
            return name
        raise RuntimeError(f"input '{name}' not found in model inputs: {self._input_names}")

    def _find_output(self, name: str, required: bool = True) -> Optional[str]:
        if name in self._output_names:
            return name
        if required:
            raise RuntimeError(f"output '{name}' not found in model outputs: {self._output_names}")
        return None

    def set_prompt(self, input_ids: List[int], attention_mask: List[int]) -> None:
        self._input_ids = np.asarray(input_ids, dtype=np.int64).reshape(1, -1)
        self._attention_mask = np.asarray(attention_mask, dtype=np.int64).reshape(1, -1)

    def infer_raw(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._input_ids is None or self._attention_mask is None:
            raise RuntimeError("Prompt not set. Call set_prompt() before inference.")
        pixel_values = preprocess_bgr(image_bgr, self.in_width, self.in_height)
        feeds = {
            self._pixel_input: pixel_values,
            self._input_ids_name: self._input_ids,
            self._attention_mask_name: self._attention_mask,
        }
        instance_masks, semantic_seg, pred_boxes, pred_logits = self.session.run_with_iobinding_numpy(feeds)
        return semantic_seg

    def infer_on_image(self, image_bgr: np.ndarray):
        semantic_seg = self.infer_raw(image_bgr)
        mask = pp.save_mask(image_bgr, semantic_seg, self._probability_threshold)
        return mask
