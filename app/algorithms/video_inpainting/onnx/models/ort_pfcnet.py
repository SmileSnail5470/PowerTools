"""
ONNX Runtime wrapper for PFCNet (Propagation Flow Completion Network).
No PyTorch dependency.
"""

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path


class ORT_PFCNet:
    """
    ONNX Runtime wrapper for PFCNet image completion.
    
    Usage:
        pfcnet = ORT_PFCNet(model_path)
        out_img = pfcnet.forward(img, mask)
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = self._get_default_model_path()
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"PFCNet ONNX model not found at {model_path}. "
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
        return Path(__file__).parent / 'onnx_models' / 'pfcnet.onnx'
    
    def forward(self, img, mask):
        """
        Forward pass through PFCNet.
        
        Args:
            img: numpy (1, 3, H, W) - masked image
            mask: numpy (1, 1, H, W) - binary mask
        
        Returns:
            out_img: numpy (1, 3, H, W) - completed image in [0, 1]
        """
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        inputs = {
            self.input_names[0]: img,
            self.input_names[1]: mask,
        }
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]
    
    def __call__(self, img, mask):
        return self.forward(img, mask)
