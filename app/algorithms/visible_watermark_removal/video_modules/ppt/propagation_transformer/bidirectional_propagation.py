import numpy as np
import onnxruntime as ort
from app.algorithms import general_inference_session, ortvalue_from_numpy, ortvalue_to_numpy
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.transformer_block import _ortvalue_to_cupy, _cupy_to_ortvalue


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
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def _to_ort(self, v):
        if isinstance(v, ort.OrtValue):
            return v
        if _HAS_CUPY and isinstance(v, cp.ndarray):
            return _cupy_to_ortvalue(v)
        return v

    def _run_backward_step(self, feat_current, feat_prop_prev, flow_prop, flow_check, mask_current):
        if self._use_iobinding:
            feed = {
                "feat_current": self._to_ort(feat_current),
                "feat_prop_prev": self._to_ort(feat_prop_prev),
                "flow_prop": self._to_ort(flow_prop),
                "flow_check": self._to_ort(flow_check),
                "mask_current": self._to_ort(mask_current),
            }
            return self.backward_step.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.backward_step.run(None, {
            "feat_current": feat_current, "feat_prop_prev": feat_prop_prev,
            "flow_prop": flow_prop, "flow_check": flow_check, "mask_current": mask_current,
        }, run_options=self.run_options)[0]

    def _run_forward_step(self, feat_current, feat_prop_prev, flow_prop, flow_check, mask_current):
        if self._use_iobinding:
            feed = {
                "feat_current": self._to_ort(feat_current),
                "feat_prop_prev": self._to_ort(feat_prop_prev),
                "flow_prop": self._to_ort(flow_prop),
                "flow_check": self._to_ort(flow_check),
                "mask_current": self._to_ort(mask_current),
            }
            return self.forward_step.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.forward_step.run(
            None, 
            {
                "feat_current": feat_current, 
                "feat_prop_prev": feat_prop_prev,
                "flow_prop": flow_prop, 
                "flow_check": flow_check, 
                "mask_current": mask_current
            }, 
            run_options=self.run_options
        )[0]

    def _run_backward_first(self, feat_current, mask_current):
        if self._use_iobinding:
            feed = {"feat_current": self._to_ort(feat_current), "mask_current": self._to_ort(mask_current)}
            return self.backward_first.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.backward_first.run(None, {"feat_current": feat_current, "mask_current": mask_current}, run_options=self.run_options)[0]

    def _run_forward_first(self, feat_current, mask_current):
        if self._use_iobinding:
            feed = {"feat_current": self._to_ort(feat_current), "mask_current": self._to_ort(mask_current)}
            return self.forward_first.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.forward_first.run(None, {"feat_current": feat_current, "mask_current": mask_current}, run_options=self.run_options)[0]

    def _run_fusion(self, outputs_b, outputs_f, mask_in, x_raw):
        if self._use_iobinding:
            feed = {
                "outputs_b": self._to_ort(outputs_b),
                "outputs_f": self._to_ort(outputs_f),
                "mask_in": self._to_ort(mask_in),
                "x_raw": self._to_ort(x_raw),
            }
            ort_out = self.fusion_sess.run_with_iobinding(feed, run_options=self.run_options)[0]
            if self._use_cupy:
                return _ortvalue_to_cupy(ort_out)
            return ortvalue_to_numpy(ort_out)
        return self.fusion_sess.run(
            None, 
            {
                "outputs_b": outputs_b, 
                "outputs_f": outputs_f,
                "mask_in": mask_in, 
                "x_raw": x_raw,
            }, 
            run_options=self.run_options
        )[0]

    def forward(self, feats, flows_forward, flows_backward, mask):
        if self._use_cupy:
            return self._forward_cupy(feats, flows_forward, flows_backward, mask)
        return self._forward_cpu(feats, flows_forward, flows_backward, mask)

    def _forward_cpu(self, feats, flows_forward, flows_backward, mask):
        b, t, c, h, w = feats.shape
        feat_list = [np.ascontiguousarray(feats[:, i]) for i in range(t)]
        mask_list = [np.ascontiguousarray(mask[:, i]) for i in range(t)]

        if self._use_iobinding:
            mask_list = [self.backward_step.to_device("mask_current", m) for m in mask_list]

        # Backward
        backward_results = [None] * t
        feat_prop_prev = ortvalue_from_numpy(np.zeros_like(feat_list[0]), use_cuda=True) if self._use_iobinding else np.zeros_like(feat_list[0])
        for i, idx in enumerate(range(t - 1, -1, -1)):
            if i == 0:
                feat_prop = self._run_backward_first(feat_list[idx], mask_list[idx])
            else:
                flow_prop = np.ascontiguousarray(flows_forward[:, idx])
                flow_check = np.ascontiguousarray(flows_backward[:, idx])
                feat_prop = self._run_backward_step(feat_list[idx], feat_prop_prev, flow_prop, flow_check, mask_list[idx])
            backward_results[idx] = feat_prop
            feat_prop_prev = feat_prop

        # Forward
        forward_results = [None] * t
        feat_prop_prev = ortvalue_from_numpy(np.zeros(backward_results[0].shape(), dtype=np.float32), use_cuda=True) if self._use_iobinding else np.zeros_like(ortvalue_to_numpy(backward_results[0]))
        for i, idx in enumerate(range(t)):
            if i == 0:
                feat_prop = self._run_forward_first(backward_results[idx], mask_list[idx])
            else:
                flow_prop = np.ascontiguousarray(flows_backward[:, i - 1])
                flow_check = np.ascontiguousarray(flows_forward[:, i - 1])
                feat_prop = self._run_forward_step(backward_results[idx], feat_prop_prev, flow_prop, flow_check, mask_list[idx])
            forward_results[idx] = feat_prop
            feat_prop_prev = feat_prop

        backward_np = [ortvalue_to_numpy(r) for r in backward_results]
        forward_np = [ortvalue_to_numpy(r) for r in forward_results]
        outputs_b = np.stack(backward_np, axis=1).reshape(-1, c, h, w)
        outputs_f = np.stack(forward_np, axis=1).reshape(-1, c, h, w)
        mask_flat = np.ascontiguousarray(mask.reshape(-1, 2, h, w))
        x_raw = np.ascontiguousarray(feats.reshape(-1, c, h, w))
        fused = self._run_fusion(outputs_b, outputs_f, mask_flat, x_raw)
        return fused.reshape(b, t, c, h, w)

    def _forward_cupy(self, feats, flows_forward, flows_backward, mask):
        b, t, c, h, w = feats.shape
        xp = cp.get_array_module(feats) if hasattr(feats, '__array_interface__') or hasattr(feats, 'device') else np
        if xp is np:
            feats_cp = cp.asarray(feats)
            flows_f_cp = cp.asarray(flows_forward)
            flows_b_cp = cp.asarray(flows_backward)
            mask_cp = cp.asarray(mask)
        else:
            feats_cp = feats
            flows_f_cp = flows_forward
            flows_b_cp = flows_backward
            mask_cp = mask

        feat_list = [cp.ascontiguousarray(feats_cp[:, i]) for i in range(t)]
        mask_list = [cp.ascontiguousarray(mask_cp[:, i]) for i in range(t)]

        # Backward propagation
        backward_results = [None] * t
        feat_prop_prev = _cupy_to_ortvalue(cp.zeros_like(feat_list[0]))
        for i, idx in enumerate(range(t - 1, -1, -1)):
            if i == 0:
                feat_prop = self._run_backward_first(feat_list[idx], mask_list[idx])
            else:
                flow_prop = cp.ascontiguousarray(flows_f_cp[:, idx])
                flow_check = cp.ascontiguousarray(flows_b_cp[:, idx])
                feat_prop = self._run_backward_step(feat_list[idx], feat_prop_prev, flow_prop, flow_check, mask_list[idx])
            backward_results[idx] = feat_prop  # OrtValue on GPU
            feat_prop_prev = feat_prop

        # Forward propagation
        forward_results = [None] * t
        feat_prop_prev = _cupy_to_ortvalue(cp.zeros(backward_results[0].shape(), dtype=cp.float32))
        for i, idx in enumerate(range(t)):
            if i == 0:
                feat_prop = self._run_forward_first(backward_results[idx], mask_list[idx])
            else:
                flow_prop = cp.ascontiguousarray(flows_b_cp[:, i - 1])
                flow_check = cp.ascontiguousarray(flows_f_cp[:, i - 1])
                feat_prop = self._run_forward_step(backward_results[idx], feat_prop_prev, flow_prop, flow_check, mask_list[idx])
            forward_results[idx] = feat_prop
            feat_prop_prev = feat_prop

        # Fusion: stack results on GPU
        backward_cp = [_ortvalue_to_cupy(r) for r in backward_results]
        forward_cp = [_ortvalue_to_cupy(r) for r in forward_results]
        outputs_b = cp.stack(backward_cp, axis=1).reshape(-1, c, h, w)
        outputs_f = cp.stack(forward_cp, axis=1).reshape(-1, c, h, w)
        del backward_results, forward_results, backward_cp, forward_cp
        mask_flat = cp.ascontiguousarray(mask_cp.reshape(-1, 2, h, w))
        x_raw = cp.ascontiguousarray(feats_cp.reshape(-1, c, h, w))
        fused = self._run_fusion(outputs_b, outputs_f, mask_flat, x_raw)
        return fused.reshape(b, t, c, h, w)

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
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def _to_ort(self, v):
        if isinstance(v, ort.OrtValue):
            return v
        if _HAS_CUPY and isinstance(v, cp.ndarray):
            return _cupy_to_ortvalue(v)
        return v

    def __call__(self, feat_current, feat_prop_prev, mask_current, mask_prop_prev, flow_prop, flow_check):
        if self._use_iobinding:
            feed = {
                self.input_names[0]: self._to_ort(feat_current),
                self.input_names[1]: self._to_ort(feat_prop_prev),
                self.input_names[2]: self._to_ort(mask_current),
                self.input_names[3]: self._to_ort(mask_prop_prev),
                self.input_names[4]: self._to_ort(flow_prop),
                self.input_names[5]: self._to_ort(flow_check),
            }
            ort_outputs = self.session.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0], ort_outputs[1]
        outputs = self.session.run(None, {
            self.input_names[0]: feat_current if not isinstance(feat_current, ort.OrtValue) else feat_current.numpy(),
            self.input_names[1]: feat_prop_prev if not isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev.numpy(),
            self.input_names[2]: mask_current if not isinstance(mask_current, ort.OrtValue) else mask_current.numpy(),
            self.input_names[3]: mask_prop_prev if not isinstance(mask_prop_prev, ort.OrtValue) else mask_prop_prev.numpy(),
            self.input_names[4]: flow_prop,
            self.input_names[5]: flow_check,
        }, run_options=self.run_options)
        return outputs[0], outputs[1]

    def __del__(self):
        self.session = None