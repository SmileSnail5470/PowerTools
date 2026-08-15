from app.algorithms import general_inference_session
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.transformer_block import _ortvalue_to_cupy, _cupy_to_ortvalue


class DecoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, sess_options=sess_options, provider_options=provider_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def __call__(self, enc_feat, run_options=None):
        run_options = self.run_options if run_options is None else run_options
        if self._use_cupy:
            feed = {self.input_names[0]: _cupy_to_ortvalue(enc_feat) if isinstance(enc_feat, cp.ndarray) else enc_feat}
            ort_outputs = self.session.run_with_iobinding(feed, run_options=run_options)
            return _ortvalue_to_cupy(ort_outputs[0])
        feed = {self.input_names[0]: enc_feat}
        if self._use_iobinding:
            return self.session.run_with_iobinding_numpy(feed, run_options=run_options)[0]
        return self.session.run(None, feed, run_options=run_options)[0]

    def __del__(self):
        self.session = None
