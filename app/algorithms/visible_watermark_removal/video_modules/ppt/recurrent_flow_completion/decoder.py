from app.algorithms import general_inference_session
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.transformer_block import _ortvalue_to_cupy, _cupy_to_ortvalue


class DecoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, feat_prop, feat_e1, x, run_options=None):
        run_options = self.run_options if run_options is None else run_options
        if self._use_cupy:
            feed = {
                "feat_prop": _cupy_to_ortvalue(feat_prop) if isinstance(feat_prop, cp.ndarray) else feat_prop,
                "feat_e1": _cupy_to_ortvalue(feat_e1) if isinstance(feat_e1, cp.ndarray) else feat_e1,
                "x": _cupy_to_ortvalue(x) if isinstance(x, cp.ndarray) else x,
            }
            ort_outputs = self.session.run_ortvalues(feed, run_options=run_options)
            return _ortvalue_to_cupy(ort_outputs[0]), _ortvalue_to_cupy(ort_outputs[1])
        feed = {"feat_prop": feat_prop, "feat_e1": feat_e1, "x": x}
        if self._use_iobinding:
            ort_outputs = self.session.run_with_iobinding_numpy(feed, run_options=run_options)
            return ort_outputs[0], ort_outputs[1]
        outputs = self.session.run(None, feed, run_options=run_options)
        return outputs[0], outputs[1]

    def __del__(self):
        self.session = None
