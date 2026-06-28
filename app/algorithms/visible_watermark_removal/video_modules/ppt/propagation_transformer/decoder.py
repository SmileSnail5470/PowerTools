import onnxruntime as ort
from app.algorithms import general_inference_session

class DecoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, sess_options=sess_options, provider_options=provider_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda

    def __call__(self, enc_feat):
        if self._use_iobinding:
            feed = {self.input_names[0]: enc_feat}
            ort_outputs = self.session.run_with_iobinding_numpy(feed, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.session.run(
            None,
            {self.input_names[0]: enc_feat},
            run_options=self.run_options
        )
        return outputs[0]
    
    def __del__(self):
        self.session = None
