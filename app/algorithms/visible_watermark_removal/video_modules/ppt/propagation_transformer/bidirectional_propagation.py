import numpy as np
import onnxruntime as ort
from app.algorithms import general_inference_session, ortvalue_from_numpy, ortvalue_to_numpy


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
        self._use_iobinding = self.backward_step.use_cuda

    def _run_backward_step(self, feat_current, feat_prop_prev, flow_prop, flow_check, mask_current):
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current if isinstance(feat_current, ort.OrtValue) else feat_current,
                "feat_prop_prev": feat_prop_prev if isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev,
                "flow_prop": flow_prop,
                "flow_check": flow_check,
                "mask_current": mask_current,
            }
            ort_outputs = self.backward_step.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
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
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current if isinstance(feat_current, ort.OrtValue) else feat_current,
                "feat_prop_prev": feat_prop_prev if isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev,
                "flow_prop": flow_prop,
                "flow_check": flow_check,
                "mask_current": mask_current,
            }
            ort_outputs = self.forward_step.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
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
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current,
                "mask_current": mask_current,
            }
            ort_outputs = self.backward_first.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
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
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current,
                "mask_current": mask_current,
            }
            ort_outputs = self.forward_first.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
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
        if self._use_iobinding:
            feed = {
                "outputs_b": outputs_b,
                "outputs_f": outputs_f,
                "mask_in": mask_in,
                "x_raw": x_raw,
            }
            ort_outputs = self.fusion_sess.run_with_iobinding(feed, run_options=self.run_options)
            return ortvalue_to_numpy(ort_outputs[0])
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
        if self._use_iobinding:
            feat_prop_prev = ortvalue_from_numpy(np.zeros_like(feat_list[0]), use_cuda=True)
        else:
            feat_prop_prev = np.zeros_like(feat_list[0])
        for i, idx in enumerate(range(T - 1, -1, -1)):
            feat_current = feat_list[idx]
            mask_current = mask_list[idx]
            if i == 0:
                feat_prop = self._run_backward_first(feat_current, mask_current)
            else:
                flow_prop = np.ascontiguousarray(flows_forward[:, idx, :, :, :])
                flow_check = np.ascontiguousarray(flows_backward[:, idx, :, :, :])
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
        if self._use_iobinding:
            feat_prop_prev = ortvalue_from_numpy(
                np.zeros(backward_results[0].shape(), dtype=np.float32), use_cuda=True
            )
        else:
            feat_prop_prev = np.zeros_like(backward_results[0])
        for i, idx in enumerate(range(T)): 
            feat_current = backward_results[idx]
            mask_current = mask_list[idx]
            if i == 0:
                feat_prop = self._run_forward_first(feat_current, mask_current)
            else:
                flow_prop = np.ascontiguousarray(flows_backward[:, i - 1, :, :, :])
                flow_check = np.ascontiguousarray(flows_forward[:, i - 1, :, :, :])
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

        backward_results = self.backward_propagation(
            feat_list, mask_list, flows_forward, flows_backward
        )

        # Forward propagation
        forward_results = self.forward_propagation(
            backward_results, mask_list, flows_forward, flows_backward
        )

        backward_np = [ortvalue_to_numpy(r) for r in backward_results]
        forward_np = [ortvalue_to_numpy(r) for r in forward_results]
        outputs_b = np.stack(backward_np, axis=1).reshape(-1, c, h, w)
        del backward_results, backward_np
        outputs_f = np.stack(forward_np, axis=1).reshape(-1, c, h, w)
        del forward_results, forward_np
        mask_flat = np.ascontiguousarray(mask.reshape(-1, 2, h, w))
        x_raw = np.ascontiguousarray(feats.reshape(-1, c, h, w))

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
        self._use_iobinding = self.session.use_cuda

    def __call__(self, feat_current, feat_prop_prev, mask_current, mask_prop_prev, flow_prop, flow_check):
        if self._use_iobinding:
            feed = {
                self.input_names[0]: feat_current if isinstance(feat_current, ort.OrtValue) else feat_current,
                self.input_names[1]: feat_prop_prev if isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev,
                self.input_names[2]: mask_current,
                self.input_names[3]: mask_prop_prev if isinstance(mask_prop_prev, ort.OrtValue) else mask_prop_prev,
                self.input_names[4]: flow_prop,
                self.input_names[5]: flow_check,
            }
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0], ort_outputs[1]
        outputs = self.session.run(
            None,
            {
                self.input_names[0]: feat_current if not isinstance(feat_current, ort.OrtValue) else feat_current.numpy(),
                self.input_names[1]: feat_prop_prev if not isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev.numpy(),
                self.input_names[2]: mask_current if not isinstance(mask_current, ort.OrtValue) else mask_current.numpy(),
                self.input_names[3]: mask_prop_prev if not isinstance(mask_prop_prev, ort.OrtValue) else mask_prop_prev.numpy(),
                self.input_names[4]: flow_prop,
                self.input_names[5]: flow_check,
            },
            run_options=self.run_options
        )
        return outputs[0], outputs[1]
    
    def __del__(self):
        self.session = None