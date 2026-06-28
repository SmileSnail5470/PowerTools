import onnxruntime as ort
from app.algorithms import general_inference_session


class DecoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda

    def __call__(self, feat_prop, feat_e1, x):
        feed = {
            "feat_prop": feat_prop,
            "feat_e1": feat_e1,
            "x": x,
        }
        if self._use_iobinding:
            ort_outputs = self.session.run_with_iobinding_numpy(feed, run_options=self.run_options)
            return ort_outputs[0], ort_outputs[1]
        outputs = self.session.run(None, feed, run_options=self.run_options)
        return outputs[0], outputs[1]
    
    def __del__(self):
        self.session = None
