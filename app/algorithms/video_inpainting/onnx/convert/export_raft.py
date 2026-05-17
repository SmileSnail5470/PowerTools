"""
Export RAFT optical flow model to ONNX.

Usage:
    python -c "from onnx.convert.export_raft import export_raft_to_onnx; export_raft_to_onnx()"
"""

import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torchvision as tv
from pathlib import Path


def export_raft_to_onnx(output_dir=None, verbose=False):
    """
    Export RAFT large model to ONNX format.
    
    The RAFT model takes two consecutive frames and outputs optical flow.
    Input shape: 2 frames normalized to [-1, 1], shape (1, 3, H, W) each
    Output shape: (1, 2, H, W)
    
    Args:
        output_dir: Directory to save the ONNX model. Defaults to onnx_models/raft
        verbose: Print export details
    """
    if output_dir is None:
        # Default: save next to the script
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        output_dir = base_dir / 'models' / 'onnx_models'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cpu'
    
    if verbose:
        print(f"Loading RAFT model...")
    
    # Load pretrained RAFT model
    raft = tv.models.optical_flow.raft_large(pretrained=True)
    raft = raft.to(device).eval()
    
    if verbose:
        print(f"RAFT model loaded.")
    
    # Export forward flow model (frame1 -> frame2)
    # RAFT forward takes (image1, image2) -> list of flow predictions
    # We'll export the full model that takes both frames
    dummy_img1 = torch.randn(1, 3, 480, 864, device=device)
    dummy_img2 = torch.randn(1, 3, 480, 864, device=device)
    
    # Define a wrapper for clean ONNX export
    class RaftWrapper(torch.nn.Module):
        def __init__(self, raft_model):
            super().__init__()
            self.raft = raft_model
        
        def forward(self, img1, img2):
            # RAFT returns list of flows, take the last one (most refined)
            flows = self.raft(img1, img2)
            return flows[-1]
    
    wrapped_raft = RaftWrapper(raft)
    
    # Export combined RAFT model
    raft_onnx_path = output_dir / 'raft_flow.onnx'
    
    if verbose:
        print(f"Exporting RAFT model to {raft_onnx_path}...")
        print(f"  Input 1: image1 (1, 3, 480, 864)")
        print(f"  Input 2: image2 (1, 3, 480, 864)")
        print(f"  Output: flow (1, 2, 480, 864)")
    
    torch.onnx.export(
        wrapped_raft,
        (dummy_img1, dummy_img2),
        str(raft_onnx_path),
        input_names=['image1', 'image2'],
        output_names=['flow'],
        dynamic_axes={
            'image1': {2: 'height', 3: 'width'},
            'image2': {2: 'height', 3: 'width'},
            'flow': {2: 'height', 3: 'width'},
        },
        opset_version=17,
        verbose=verbose,
    )
    
    if verbose:
        print(f"RAFT model exported successfully to {raft_onnx_path}")
        print(f"File size: {raft_onnx_path.stat().st_size / (1024*1024):.1f} MB")
    
    return str(raft_onnx_path)


if __name__ == '__main__':
    export_raft_to_onnx(verbose=True)
