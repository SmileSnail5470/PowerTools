import numpy as np
import onnxruntime as ort
from app.algorithms import general_inference_session
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.transformer_block import _ortvalue_to_cupy, _cupy_to_ortvalue


class EncoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, masked_flows, masks):
        if self._use_cupy:
            feed = {
                "masked_flows": _cupy_to_ortvalue(masked_flows) if isinstance(masked_flows, cp.ndarray) else masked_flows,
                "masks": _cupy_to_ortvalue(masks) if isinstance(masks, cp.ndarray) else masks,
            }
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return _ortvalue_to_cupy(ort_outputs[0]), _ortvalue_to_cupy(ort_outputs[1]), _ortvalue_to_cupy(ort_outputs[2])
        feed = {"masked_flows": masked_flows, "masks": masks}
        if self._use_iobinding:
            ort_outputs = self.session.run_with_iobinding_numpy(feed, run_options=self.run_options)
            return ort_outputs[0], ort_outputs[1], ort_outputs[2]
        outputs = self.session.run(None, feed, run_options=self.run_options)
        return outputs[0], outputs[1], outputs[2]

    def __del__(self):
        self.session = None
