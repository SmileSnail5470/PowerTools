import numpy as np
import onnxruntime as ort
from app.algorithms import general_inference_session, ortvalue_from_numpy, ortvalue_to_numpy


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
        self.backward_session = general_inference_session(
            backward_onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
        )
        self.forward_session = general_inference_session(
            forward_onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
        )
        self.backward_backbone_session = general_inference_session(
            backward_backbone_onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
        )
        self.forward_backbone_session = general_inference_session(
            forward_backbone_onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
        )
        self.fusion_session = general_inference_session(
            fusion_onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
        )
        self.run_options = run_options
        self._use_iobinding = self.backward_session.use_cuda

    def run_backward_step(self, feat_current, feat_prop_prev, feat_n2):
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev if isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev,
                "feat_n2": feat_n2 if isinstance(feat_n2, ort.OrtValue) else feat_n2,
            }
            ort_outputs = self.backward_session.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.backward_session.run(
            None,
            {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev,
                "feat_n2": feat_n2,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def run_forward_step(self, feat_current, feat_prop_prev, feat_n2, feat_backward):
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev if isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev,
                "feat_n2": feat_n2 if isinstance(feat_n2, ort.OrtValue) else feat_n2,
                "feat_backward": feat_backward if isinstance(feat_backward, ort.OrtValue) else feat_backward,
            }
            ort_outputs = self.forward_session.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.forward_session.run(
            None,
            {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev,
                "feat_n2": feat_n2,
                "feat_backward": feat_backward,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def run_backward_backbone(self, feat_current, feat_prop_prev):
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev if isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev,
            }
            ort_outputs = self.backward_backbone_session.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.backward_backbone_session.run(
            None,
            {
                "feat_current": feat_current,
                "feat_prop_prev": feat_prop_prev,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def run_forward_backbone(self, feat_current, feat_backward, feat_prop_prev):
        if self._use_iobinding:
            feed = {
                "feat_current": feat_current,
                "feat_backward": feat_backward if isinstance(feat_backward, ort.OrtValue) else feat_backward,
                "feat_prop_prev": feat_prop_prev if isinstance(feat_prop_prev, ort.OrtValue) else feat_prop_prev,
            }
            ort_outputs = self.forward_backbone_session.run_with_iobinding(feed, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.forward_backbone_session.run(
            None,
            {
                "feat_current": feat_current,
                "feat_backward": feat_backward,
                "feat_prop_prev": feat_prop_prev,
            },
            run_options=self.run_options
        )
        return outputs[0]

    def fusion_conv(self, x):
        if self._use_iobinding:
            feed = {"x": x}
            ort_outputs = self.fusion_session.run_with_iobinding_numpy(feed, run_options=self.run_options)
            return ort_outputs[0]
        outputs = self.fusion_session.run(
            None,
            {"x": x},
            run_options=self.run_options
        )
        return outputs[0]

    def backward_propagation(self, feat_list):
        T = len(feat_list)
        backward_results = [None] * T
        
        feat_n2_prev = None
        if self._use_iobinding:
            feat_prop_prev = ortvalue_from_numpy(np.zeros_like(feat_list[0]), use_cuda=True)
        else:
            feat_prop_prev = np.zeros_like(feat_list[0])
        
        for i, idx in enumerate(range(T - 1, -1, -1)):
            feat_current = feat_list[idx]
            if i == 0:
                feat_prop = self.run_backward_backbone(feat_current, feat_prop_prev)
            else:
                if i > 1:
                    feat_n2 = feat_n2_prev
                else:
                    if self._use_iobinding:
                        feat_n2 = ortvalue_from_numpy(
                            np.zeros(feat_prop_prev.shape(), dtype=np.float32), use_cuda=True
                        )
                    else:
                        feat_n2 = np.zeros_like(feat_prop_prev)
                feat_prop = self.run_backward_step(feat_current, feat_prop_prev, feat_n2)
            
            backward_results[idx] = ortvalue_to_numpy(feat_prop)
            feat_n2_prev = feat_prop_prev
            feat_prop_prev = feat_prop
            
        return backward_results

    def forward_propagation(self, feat_list, backward_results):
        T = len(feat_list)
        forward_results = [None] * T
        
        feat_n2_prev = None
        if self._use_iobinding:
            feat_prop_prev = ortvalue_from_numpy(np.zeros_like(feat_list[0]), use_cuda=True)
        else:
            feat_prop_prev = np.zeros_like(feat_list[0])
        
        for i, idx in enumerate(range(T)):
            feat_current = feat_list[idx]
            if i == 0:
                feat_prop = self.run_forward_backbone(feat_current, backward_results[idx], feat_prop_prev)
            else:
                if i > 1:
                    feat_n2 = feat_n2_prev
                else:
                    if self._use_iobinding:
                        feat_n2 = ortvalue_from_numpy(
                            np.zeros(feat_prop_prev.shape(), dtype=np.float32), use_cuda=True
                        )
                    else:
                        feat_n2 = np.zeros_like(feat_prop_prev)
                feat_prop = self.run_forward_step(feat_current, feat_prop_prev, feat_n2, backward_results[idx])
                
            forward_results[idx] = ortvalue_to_numpy(feat_prop)
            feat_n2_prev = feat_prop_prev
            feat_prop_prev = feat_prop
            
        return forward_results

    def forward(self, feats):
        B, T, C, H, W = feats.shape
        feat_list = [np.ascontiguousarray(feats[:, i]) for i in range(T)]
        
        backward_results = self.backward_propagation(feat_list)
        forward_results = self.forward_propagation(feat_list, backward_results)
        
        outputs = []
        for i in range(T):
            fused = np.ascontiguousarray(np.concatenate([backward_results[i], forward_results[i]], axis=1))
            fused = self.fusion_conv(fused)
            outputs.append(fused)
        
        del backward_results, forward_results
        outputs = np.stack(outputs, axis=1)
        outputs = outputs + feats
        return outputs
    
    def __del__(self):
        self.backward_session = None
        self.forward_session = None
        self.backward_backbone_session = None
        self.forward_backbone_session = None
        self.fusion_session = None