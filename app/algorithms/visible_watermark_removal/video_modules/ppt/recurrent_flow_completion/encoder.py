import onnxruntime as ort
from app.algorithms import general_inference_session


class EncoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda

    def __call__(self, masked_flows, masks):
        feed = {
            "masked_flows": masked_flows,
            "masks": masks,
        }
        if self._use_iobinding:
            ort_outputs = self.session.run_with_iobinding_numpy(feed, run_options=self.run_options)
            return ort_outputs[0], ort_outputs[1], ort_outputs[2]
        outputs = self.session.run(None, feed, run_options=self.run_options)
        return outputs[0], outputs[1], outputs[2]
    
    def __del__(self):
        self.session = None
