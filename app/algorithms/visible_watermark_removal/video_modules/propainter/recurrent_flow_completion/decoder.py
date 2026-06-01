import onnxruntime as ort


class DecoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = ort.InferenceSession(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.run_options = run_options

    def __call__(self, feat_prop, feat_e1, x):
        outputs = self.session.run(
            None,
            {
                "feat_prop": feat_prop,
                "feat_e1": feat_e1,
                "x": x,
            },
            run_options=self.run_options
        )
        return outputs[0], outputs[1]
    
    def __del__(self):
        del self.session
