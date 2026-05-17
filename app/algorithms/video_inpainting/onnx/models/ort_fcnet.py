"""
ONNX Runtime wrapper for FCNet (Flow Completion Network).
No PyTorch dependency.
"""

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path


class ORT_FCNet:
    """
    ONNX Runtime wrapper for FCNet flow completion.
    
    Usage:
        fcnet = ORT_FCNet(model_path)
        flows = fcnet.forward(masked_flows, masks)
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = self._get_default_model_path()
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"FCNet ONNX model not found at {model_path}. "
                f"Please run the export script first."
            )
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = True
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def _get_default_model_path(self):
        return Path(__file__).parent / 'onnx_models' / 'fcnet.onnx'
    
    def forward(self, masked_flows, masks):
        """
        Forward pass through FCNet.
        
        Args:
            masked_flows: numpy (1, T, 2, 240, 432) - masked optical flows
            masks: numpy (1, T, 1, 240, 432) - flow masks
        
        Returns:
            completed_flows: numpy (1, T, 2, 240, 432)
        """
        if masked_flows.dtype != np.float32:
            masked_flows = masked_flows.astype(np.float32)
        if masks.dtype != np.float32:
            masks = masks.astype(np.float32)
        
        inputs = {
            self.input_names[0]: masked_flows,
            self.input_names[1]: masks,
        }
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]
    
    def __call__(self, masked_flows, masks):
        return self.forward(masked_flows, masks)
    
    def forward_bidirect_flow(self, masked_flows_bi, masks):
        """
        Bidirectional flow completion - mirrors FCNet.forward_bidirect_flow.
        
        Args:
            masked_flows_bi: list of 2 numpy arrays [(1, T, 2, 240, 432), (1, T, 2, 240, 432)]
                             forward and backward masked flows
            masks: numpy (1, L, 1, 240, 432) - frame masks where L = T + 1
        
        Returns:
            (pred_forward, pred_backward): tuple of 2 numpy arrays
        """
        T = masked_flows_bi[0].shape[1]
        
        masks_forward = masks[:, :-1, ...].copy()
        masks_backward = masks[:, 1:, ...].copy()
        
        masked_flows_forward = masked_flows_bi[0] * (1 - masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1 - masks_backward)
        
        pred_flows_forward = self.forward(masked_flows_forward, masks_forward)
        
        masked_flows_backward = np.flip(masked_flows_backward, axis=1)
        masks_backward = np.flip(masks_backward, axis=1)
        pred_flows_backward = self.forward(masked_flows_backward, masks_backward)
        pred_flows_backward = np.flip(pred_flows_backward, axis=1)
        
        return pred_flows_forward, pred_flows_backward
    
    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        """
        Combine original flows with predicted flows - mirrors FCNet.combine_flow.
        """
        masks_forward = masks[:, :-1, ...].copy()
        masks_backward = masks[:, 1:, ...].copy()
        
        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1 - masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1 - masks_backward)
        
        return pred_flows_forward, pred_flows_backward
