"""
ONNX Runtime wrapper for RAFT optical flow model.
No PyTorch dependency.
"""

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path


class ORT_Raft:
    """
    ONNX Runtime wrapper for RAFT optical flow model.
    
    Usage:
        raft = ORT_Raft(model_path)
        flow = raft.forward(img1, img2)  # each: (1, 3, H, W) numpy in [-1, 1]
    """
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path to the ONNX model file. If None, uses default path.
        """
        if model_path is None:
            model_path = self._get_default_model_path()
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RAFT ONNX model not found at {model_path}. "
                                    f"Please run the export script first.")
        
        # Create ONNX Runtime session
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
        """Get default model path relative to this file."""
        return Path(__file__).parent / 'onnx_models' / 'raft_flow.onnx'
    
    def forward(self, image1, image2):
        """
        Compute optical flow between two frames.
        
        Args:
            image1: numpy array (1, 3, H, W), values in [-1, 1]
            image2: numpy array (1, 3, H, W), values in [-1, 1]
        
        Returns:
            flow: numpy array (1, 2, H, W)
        """
        # Ensure float32
        if image1.dtype != np.float32:
            image1 = image1.astype(np.float32)
        if image2.dtype != np.float32:
            image2 = image2.astype(np.float32)
        
        inputs = {
            self.input_names[0]: image1,
            self.input_names[1]: image2,
        }
        
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]
    
    def __call__(self, image1, image2):
        return self.forward(image1, image2)
