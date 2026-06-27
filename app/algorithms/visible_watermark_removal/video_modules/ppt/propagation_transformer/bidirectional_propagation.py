import numpy as np
import onnxruntime as ort
from app.algorithms import general_inference_session

class BidirectionalPropagationORT:
    def __init__(
        self,
        backward_step_path,
        forward_step_path,
        backward_first_path,
        forward_first_path,
        fusion_path,
        providers,
        provider_options=None, 
        sess_options=None, 
        run_options=None
    ):
        self.backward_step = general_inference_session(backward_step_path, providers=providers, sess_options=sess_options, provider_options=provider_options)
        self.forward_step = general_inference_session(forward_step_path, providers=providers, sess_options=sess_options, provider_options=provider_options)
        self.backward_first = general_inference_session(backward_first_path, providers=providers, sess_options=sess_options, provider_options=provider_options)
        self.forward_first = general_inference_session(forward_first_path, providers=providers, sess_options=sess_options, provider_options=provider_options)
        self.fusion_sess = general_inference_session(fusion_path, providers=providers, sess_options=sess_options, provider_options=provider_options)
        self.run_options = run_options

    def _run_backward_step(self, feat_current, feat_prop_prev, flow_prop, flow_check, mask_current):
        outputs = self.backward_step.run(
            None,
            {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev,
                "flow_prop": flow_prop,
                "flow_check": flow_check,
                "mask_current": mask_current,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def _run_forward_step(self, feat_current, feat_prop_prev, flow_prop, flow_check, mask_current):
        outputs = self.forward_step.run(
            None,
            {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev,
                "flow_prop": flow_prop,
                "flow_check": flow_check,
                "mask_current": mask_current,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def _run_backward_first(self, feat_current, mask_current):
        outputs = self.backward_first.run(
            None,
            {
                "feat_current": feat_current,
                "mask_current": mask_current,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def _run_forward_first(self, feat_current, mask_current):
        outputs = self.forward_first.run(
            None,
            {
                "feat_current": feat_current,
                "mask_current": mask_current,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def _run_fusion(self, outputs_b, outputs_f, mask_in, x_raw):
        outputs = self.fusion_sess.run(
            None,
            {
                "outputs_b": outputs_b,
                "outputs_f": outputs_f,
                "mask_in": mask_in,
                "x_raw": x_raw,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def backward_propagation(self, feat_list, mask_list, flows_forward, flows_backward):
        T = len(feat_list)
        results = [None] * T
        feat_prop_prev = np.zeros_like(feat_list[0])
        for i, idx in enumerate(range(T - 1, -1, -1)):
            feat_current = feat_list[idx]
            mask_current = mask_list[idx]
            if i == 0:
                feat_prop = self._run_backward_first(feat_current, mask_current)
            else:
                flow_prop = flows_forward[:, idx, :, :, :]
                flow_check = flows_backward[:, idx, :, :, :]
                feat_prop = self._run_backward_step(
                    feat_current, feat_prop_prev,
                    flow_prop, flow_check,
                    mask_current
                )
            results[idx] = feat_prop
            feat_prop_prev = feat_prop
        return results

    def forward_propagation(self, backward_results, mask_list, flows_forward, flows_backward):
        T = len(backward_results)
        results = [None] * T
        feat_prop_prev = np.zeros_like(backward_results[0])
        for i, idx in enumerate(range(T)): 
            feat_current = backward_results[idx]
            mask_current = mask_list[idx]
            if i == 0:
                feat_prop = self._run_forward_first(feat_current, mask_current)
            else:
                flow_prop = flows_backward[:, i - 1, :, :, :]
                flow_check = flows_forward[:, i - 1, :, :, :]
                feat_prop = self._run_forward_step(
                    feat_current, feat_prop_prev,
                    flow_prop, flow_check,
                    mask_current
                )
            results[idx] = feat_prop
            feat_prop_prev = feat_prop
        return results

    def forward(self, feats, flows_forward, flows_backward, mask):
        b, t, c, h, w = feats.shape
        feat_list = [np.ascontiguousarray(feats[:, i]) for i in range(t)]
        mask_list = [np.ascontiguousarray(mask[:, i]) for i in range(t)]

        # Backward propagation
        backward_results = self.backward_propagation(
            feat_list, mask_list, flows_forward, flows_backward
        )

        # Forward propagation
        forward_results = self.forward_propagation(
            backward_results, mask_list, flows_forward, flows_backward
        )

        # Stack results and fuse
        outputs_b = np.stack(backward_results, axis=1).reshape(-1, c, h, w)
        del backward_results
        outputs_f = np.stack(forward_results, axis=1).reshape(-1, c, h, w)
        del forward_results
        mask_flat = mask.reshape(-1, 2, h, w)
        x_raw = feats.reshape(-1, c, h, w)

        fused = self._run_fusion(outputs_b, outputs_f, mask_flat, x_raw)
        del outputs_b, outputs_f, mask_flat, x_raw
        fused = fused.reshape(b, t, c, h, w)
        return fused
    
    def __del__(self):
        self.backward_step = None
        self.forward_step = None
        self.backward_first = None
        self.forward_first = None
        self.fusion_sess = None


class ImgPropStepORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options

    def __call__(self, feat_current, feat_prop_prev, mask_current, mask_prop_prev, flow_prop, flow_check):
        outputs = self.session.run(
            None,
            {
                self.input_names[0]: feat_current,
                self.input_names[1]: feat_prop_prev,
                self.input_names[2]: mask_current,
                self.input_names[3]: mask_prop_prev,
                self.input_names[4]: flow_prop,
                self.input_names[5]: flow_check,
            },
            run_options=self.run_options
        )
        return outputs[0], outputs[1]
    
    def __del__(self):
        self.session = None