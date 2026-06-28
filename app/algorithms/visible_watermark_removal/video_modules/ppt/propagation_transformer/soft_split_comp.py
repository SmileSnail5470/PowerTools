import onnxruntime as ort
from app.algorithms import general_inference_session


class SoftSplitORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda

    def __call__(self, x):
        feed_dict = {
            self.input_names[0]: x,
        }
        if self._use_iobinding:
            ort_outputs = self.session.run_with_iobinding_numpy(feed_dict, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.session.run(None, feed_dict, run_options=self.run_options)
        return outputs[0]
    
    def __del__(self):
        self.session = None


class SoftCompORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options
        self._use_iobinding = self.session.use_cuda

    def __call__(self, x, enc_feat):
        feed_dict = {
            self.input_names[0]: x,
            self.input_names[1]: enc_feat,
        }
        if self._use_iobinding:
            ort_outputs = self.session.run_with_iobinding_numpy(feed_dict, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.session.run(None, feed_dict, run_options=self.run_options)
        return outputs[0]
    
    def __del__(self):
        self.session = None
