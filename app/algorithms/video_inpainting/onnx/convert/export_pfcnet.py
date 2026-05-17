"""
Export PFCNet (Propagation Flow Completion Network) to ONNX format.

PFCNet is a standard encoder-decoder with gated convolutions.
Being pure PyTorch without custom ops, it exports directly.

Usage:
    python -m onnx.convert.export_pfcnet
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
from pathlib import Path


def export_pfcnet_to_onnx(output_dir=None, verbose=False):
    """
    Export PFCNet to ONNX format.

    PFCNet inputs:
        img: (1, 3, H, W) - masked image
        mask: (1, 1, H, W) - binary mask
    PFCNet output:
        out_img: (1, 3, H, W) - completed image in [0, 1]
    """
    if output_dir is None:
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        output_dir = base_dir / 'models' / 'onnx_models'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cpu'

    if verbose:
        print("Loading PFCNet model...")

    from pfcnet import PFCNet
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'weights', 'PFCNet.pth'
    )
    pfcnet = PFCNet(weights_path).to(device).eval()
    if verbose:
        print(f"PFCNet model loaded from {weights_path}")

    H, W = 480, 864
    dummy_img = torch.randn(1, 3, H, W, device=device)
    dummy_mask = torch.randint(0, 2, (1, 1, H, W), device=device).float()
    pfcnet_onnx_path = output_dir / 'pfcnet.onnx'

    if verbose:
        print(f"Exporting PFCNet to {pfcnet_onnx_path}...")
    torch.onnx.export(
        pfcnet, (dummy_img, dummy_mask), str(pfcnet_onnx_path),
        input_names=['img', 'mask'],
        output_names=['out_img'],
        dynamic_axes={
            'img': {2: 'height', 3: 'width'},
            'mask': {2: 'height', 3: 'width'},
            'out_img': {2: 'height', 3: 'width'},
        },
        opset_version=17, verbose=verbose,
    )
    if verbose:
        size_mb = pfcnet_onnx_path.stat().st_size / (1024 * 1024)
        print(f"PFCNet exported to {pfcnet_onnx_path} ({size_mb:.1f} MB)")
    return str(pfcnet_onnx_path)


if __name__ == '__main__':
    export_pfcnet_to_onnx(verbose=True)
