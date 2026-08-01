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
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, masked_frames, masks_in, masks_updated):
        if self._use_cupy:
            feed = {
                self.input_names[0]: _cupy_to_ortvalue(masked_frames) if isinstance(masked_frames, cp.ndarray) else masked_frames,
                self.input_names[1]: _cupy_to_ortvalue(masks_in) if isinstance(masks_in, cp.ndarray) else masks_in,
                self.input_names[2]: _cupy_to_ortvalue(masks_updated) if isinstance(masks_updated, cp.ndarray) else masks_updated,
            }
            ort_outputs = self.session.run_ortvalues(feed, run_options=self.run_options)
            return _ortvalue_to_cupy(ort_outputs[0])
        feed_dict = {
            self.input_names[0]: masked_frames,
            self.input_names[1]: masks_in,
            self.input_names[2]: masks_updated,
        }
        if self._use_iobinding:
            return self.session.run_with_iobinding_numpy(feed_dict, run_options=self.run_options)[0]
        return self.session.run(None, feed_dict, run_options=self.run_options)[0]

    def __del__(self):
        self.session = None
