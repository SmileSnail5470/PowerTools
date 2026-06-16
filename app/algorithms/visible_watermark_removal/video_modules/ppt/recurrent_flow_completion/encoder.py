import onnxruntime as ort
from app.algorithms import general_inference_session


class EncoderORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.run_options = run_options

    def __call__(self, masked_flows, masks):
        outputs = self.session.run(
            None,
            {
                "masked_flows": masked_flows,
                "masks": masks,
            },
            run_options=self.run_options
        )
        return outputs[0], outputs[1], outputs[2]
    
    def __del__(self):
        del self.session
