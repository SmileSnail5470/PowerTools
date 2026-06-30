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
            backward_onnx_path, 
            forward_onnx_path,
            backward_backbone_onnx_path, 
            forward_backbone_onnx_path,
            fusion_onnx_path, 
            providers, 
            provider_options=None,
            sess_options=None, 
            run_options=None
        ):
        self.backward_session = general_inference_session(backward_onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.forward_session = general_inference_session(forward_onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.backward_backbone_session = general_inference_session(backward_backbone_onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.forward_backbone_session = general_inference_session(forward_backbone_onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.fusion_session = general_inference_session(fusion_onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.run_options = run_options
        self._use_iobinding = self.backward_session.use_cuda
        self._use_cupy = self._use_iobinding and _HAS_CUPY

    def _to_ort(self, v):
        if isinstance(v, ort.OrtValue):
            return v
        if _HAS_CUPY and isinstance(v, cp.ndarray):
            return _cupy_to_ortvalue(v)
        return v

    def run_backward_step(self, feat_current, feat_prop_prev, feat_n2):
        if self._use_iobinding:
            feed = {"feat_current": self._to_ort(feat_current), "feat_prop_prev": self._to_ort(feat_prop_prev), "feat_n2": self._to_ort(feat_n2)}
            return self.backward_session.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.backward_session.run(None, {"feat_current": feat_current, "feat_prop_prev": feat_prop_prev, "feat_n2": feat_n2}, run_options=self.run_options)[0]

    def run_forward_step(self, feat_current, feat_prop_prev, feat_n2, feat_backward):
        if self._use_iobinding:
            feed = {"feat_current": self._to_ort(feat_current), "feat_prop_prev": self._to_ort(feat_prop_prev), "feat_n2": self._to_ort(feat_n2), "feat_backward": self._to_ort(feat_backward)}
            return self.forward_session.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.forward_session.run(None, {"feat_current": feat_current, "feat_prop_prev": feat_prop_prev, "feat_n2": feat_n2, "feat_backward": feat_backward}, run_options=self.run_options)[0]

    def run_backward_backbone(self, feat_current, feat_prop_prev):
        if self._use_iobinding:
            feed = {"feat_current": self._to_ort(feat_current), "feat_prop_prev": self._to_ort(feat_prop_prev)}
            return self.backward_backbone_session.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.backward_backbone_session.run(None, {"feat_current": feat_current, "feat_prop_prev": feat_prop_prev}, run_options=self.run_options)[0]

    def run_forward_backbone(self, feat_current, feat_backward, feat_prop_prev):
        if self._use_iobinding:
            feed = {"feat_current": self._to_ort(feat_current), "feat_backward": self._to_ort(feat_backward), "feat_prop_prev": self._to_ort(feat_prop_prev)}
            return self.forward_backbone_session.run_with_iobinding(feed, run_options=self.run_options)[0]
        return self.forward_backbone_session.run(None, {"feat_current": feat_current, "feat_backward": feat_backward, "feat_prop_prev": feat_prop_prev}, run_options=self.run_options)[0]

    def fusion_conv(self, x):
        if self._use_cupy:
            feed = {"x": _cupy_to_ortvalue(x) if isinstance(x, cp.ndarray) else x}
            ort_out = self.fusion_session.run_with_iobinding(feed, run_options=self.run_options)[0]
            return _ortvalue_to_cupy(ort_out)
        if self._use_iobinding:
            return self.fusion_session.run_with_iobinding_numpy({"x": x}, run_options=self.run_options)[0]
        return self.fusion_session.run(None, {"x": x}, run_options=self.run_options)[0]

    def forward(self, feats):
        if self._use_cupy:
            return self._forward_cupy(feats)
        return self._forward_cpu(feats)

    def _forward_cpu(self, feats):
        B, T, C, H, W = feats.shape
        feat_list = [np.ascontiguousarray(feats[:, i]) for i in range(T)]

        # Backward
        backward_results = [None] * T
        feat_prop_prev = ortvalue_from_numpy(np.zeros_like(feat_list[0]), use_cuda=True) if self._use_iobinding else np.zeros_like(feat_list[0])
        feat_n2_prev = None
        for i, idx in enumerate(range(T - 1, -1, -1)):
            if i == 0:
                feat_prop = self.run_backward_backbone(feat_list[idx], feat_prop_prev)
            else:
                feat_n2 = feat_n2_prev if i > 1 else (ortvalue_from_numpy(np.zeros(feat_prop_prev.shape(), dtype=np.float32), use_cuda=True) if self._use_iobinding else np.zeros_like(feat_prop_prev))
                feat_prop = self.run_backward_step(feat_list[idx], feat_prop_prev, feat_n2)
            backward_results[idx] = ortvalue_to_numpy(feat_prop)
            feat_n2_prev = feat_prop_prev
            feat_prop_prev = feat_prop

        # Forward
        forward_results = [None] * T
        feat_prop_prev = ortvalue_from_numpy(np.zeros_like(feat_list[0]), use_cuda=True) if self._use_iobinding else np.zeros_like(feat_list[0])
        feat_n2_prev = None
        for i, idx in enumerate(range(T)):
            if i == 0:
                feat_prop = self.run_forward_backbone(feat_list[idx], backward_results[idx], feat_prop_prev)
            else:
                feat_n2 = feat_n2_prev if i > 1 else (ortvalue_from_numpy(np.zeros(feat_prop_prev.shape(), dtype=np.float32), use_cuda=True) if self._use_iobinding else np.zeros_like(feat_prop_prev))
                feat_prop = self.run_forward_step(feat_list[idx], feat_prop_prev, feat_n2, backward_results[idx])
            forward_results[idx] = ortvalue_to_numpy(feat_prop)
            feat_n2_prev = feat_prop_prev
            feat_prop_prev = feat_prop

        # Fusion
        outputs = []
        for i in range(T):
            fused = np.ascontiguousarray(np.concatenate([backward_results[i], forward_results[i]], axis=1))
            outputs.append(self.fusion_conv(fused))
        outputs = np.stack(outputs, axis=1)
        return outputs + feats

    def _forward_cupy(self, feats):
        B, T, C, H, W = feats.shape
        feats_cp = cp.asarray(feats) if not isinstance(feats, cp.ndarray) else feats
        feat_list = [cp.ascontiguousarray(feats_cp[:, i]) for i in range(T)]

        # Backward
        backward_ort = [None] * T
        feat_prop_prev = _cupy_to_ortvalue(cp.zeros_like(feat_list[0]))
        feat_n2_prev = None
        for i, idx in enumerate(range(T - 1, -1, -1)):
            if i == 0:
                feat_prop = self.run_backward_backbone(feat_list[idx], feat_prop_prev)
            else:
                feat_n2 = feat_n2_prev if i > 1 else _cupy_to_ortvalue(cp.zeros(feat_prop_prev.shape(), dtype=cp.float32))
                feat_prop = self.run_backward_step(feat_list[idx], feat_prop_prev, feat_n2)
            backward_ort[idx] = feat_prop
            feat_n2_prev = feat_prop_prev
            feat_prop_prev = feat_prop

        # Forward
        forward_ort = [None] * T
        feat_prop_prev = _cupy_to_ortvalue(cp.zeros_like(feat_list[0]))
        feat_n2_prev = None
        for i, idx in enumerate(range(T)):
            if i == 0:
                feat_prop = self.run_forward_backbone(feat_list[idx], backward_ort[idx], feat_prop_prev)
            else:
                feat_n2 = feat_n2_prev if i > 1 else _cupy_to_ortvalue(cp.zeros(feat_prop_prev.shape(), dtype=cp.float32))
                feat_prop = self.run_forward_step(feat_list[idx], feat_prop_prev, feat_n2, backward_ort[idx])
            forward_ort[idx] = feat_prop
            feat_n2_prev = feat_prop_prev
            feat_prop_prev = feat_prop

        # Fusion on GPU
        backward_cp = [_ortvalue_to_cupy(r) for r in backward_ort]
        forward_cp = [_ortvalue_to_cupy(r) for r in forward_ort]
        outputs = []
        for i in range(T):
            fused = cp.ascontiguousarray(cp.concatenate([backward_cp[i], forward_cp[i]], axis=1))
            outputs.append(self.fusion_conv(fused))
        outputs = cp.stack(outputs, axis=1)
        return cp.asnumpy(outputs + feats_cp)

    def __del__(self):
        self.backward_session = None
        self.forward_session = None
        self.backward_backbone_session = None
        self.forward_backbone_session = None
        self.fusion_session = None