"""
Export FCNet (Flow Completion Network) to ONNX format.

FCNet uses ModulatedDeformConv2d from mmcv (custom CUDA op).
For ONNX export, we replace it with a pure-PyTorch equivalent using grid_sample
decomposition that torch.onnx.export can trace.

Usage:
    python -m onnx.convert.export_fcnet
"""

import os
import sys
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def modulated_deform_conv2d_export(
    x, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups
):
    """
    Pure PyTorch implementation of modulated_deform_conv2d that is ONNX-exportable.
    Decomposes deformable convolution per-kernel-element using grid_sample.
    """
    N, C_in, H, W = x.shape
    C_out, Cg, kH, kW = weight.shape
    assert kH == 3 and kW == 3, "Only 3x3 kernel supported"
    assert stride == 1 and padding == 1

    group_in_channels = C_in // deform_groups
    output = torch.zeros(N, C_out, H, W, device=x.device, dtype=x.dtype)

    base_y = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
    base_x = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(base_y, base_x, indexing='ij')
    dx_pixel = 2.0 / (W - 1)
    dy_pixel = 2.0 / (H - 1)
    kernel_pos = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),  (0, 0),  (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

    for g in range(deform_groups):
        c_start = g * group_in_channels
        c_end = (g + 1) * group_in_channels
        x_g = x[:, c_start:c_end, :, :]
        off_g = offset[:, g * 18:(g + 1) * 18, :, :]
        mask_g = mask[:, g * 9:(g + 1) * 9, :, :]
        w_g = weight[:, c_start:c_end, :, :]

        for ki, (ky, kx) in enumerate(kernel_pos):
            off_dx = off_g[:, ki * 2:ki * 2 + 1, :, :]
            off_dy = off_g[:, ki * 2 + 1:ki * 2 + 2, :, :]
            mod = mask_g[:, ki:ki + 1, :, :]
            sample_x = gx[None, None, :, :] + kx * dx_pixel + off_dx * dx_pixel
            sample_y = gy[None, None, :, :] + ky * dy_pixel + off_dy * dy_pixel
            grid = torch.cat([
                sample_x.permute(0, 2, 3, 1),
                sample_y.permute(0, 2, 3, 1)
            ], dim=-1)
            sampled = F.grid_sample(x_g, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled = sampled * mod
            w_ki = w_g[:, :, ky + 1, kx + 1]
            for co in range(C_out):
                output[:, co:co + 1, :, :] = output[:, co:co + 1, :, :] + \
                    (sampled * w_ki[co:co + 1, :, None, None]).sum(dim=1, keepdim=True)

    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


def export_fcnet_to_onnx(output_dir=None, verbose=False):
    """
    Export FCNet to ONNX format.

    FCNet inputs:
        masked_flows: (1, T, 2, 240, 432)
        masks: (1, T, 1, 240, 432)
    FCNet output:
        completed_flows: (1, T, 2, 240, 432)
    """
    if output_dir is None:
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        output_dir = base_dir / 'models' / 'onnx_models'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cpu'

    if verbose:
        print("Loading FCNet model and patching for ONNX export...")

    # Monkey-patch mmcv modules before fcnet module-level code runs
    import fcnet as fcnet_module

    class PatchedModulatedDeformConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     dilation=1, groups=1, deform_groups=1, bias=True, **kwargs):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            k = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
            self.kernel_size = k
            self.stride = stride if isinstance(stride, (tuple, list)) else (stride, stride)
            self.padding = padding if isinstance(padding, (tuple, list)) else (padding, padding)
            self.dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
            self.groups = groups
            self.deform_groups = deform_groups
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *k))
            self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        def forward(self, x, offset, mask):
            return modulated_deform_conv2d_export(
                x, offset, mask, self.weight, self.bias,
                self.stride, self.padding, self.dilation,
                self.groups, self.deform_groups
            )

    fcnet_module.ModulatedDeformConv2d = PatchedModulatedDeformConv2d
    fcnet_module.modulated_deform_conv2d = modulated_deform_conv2d_export
    importlib.reload(fcnet_module)

    from fcnet import FCNet
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'weights', 'FCNet.pth'
    )
    fcnet = FCNet(weights_path).to(device).eval()
    if verbose:
        print(f"FCNet model loaded from {weights_path}")

    T, H, W = 5, 240, 432
    dummy_flows = torch.randn(1, T, 2, H, W, device=device)
    dummy_masks = torch.randn(1, T, 1, H, W, device=device)
    fcnet_onnx_path = output_dir / 'fcnet.onnx'

    if verbose:
        print(f"Exporting FCNet to {fcnet_onnx_path}...")
    torch.onnx.export(
        fcnet, (dummy_flows, dummy_masks), str(fcnet_onnx_path),
        input_names=['masked_flows', 'masks'],
        output_names=['completed_flows'],
        dynamic_axes={
            'masked_flows': {1: 'temporal'},
            'masks': {1: 'temporal'},
            'completed_flows': {1: 'temporal'},
        },
        opset_version=17, verbose=verbose,
    )
    if verbose:
        size_mb = fcnet_onnx_path.stat().st_size / (1024 * 1024)
        print(f"FCNet exported to {fcnet_onnx_path} ({size_mb:.1f} MB)")
    return str(fcnet_onnx_path)


if __name__ == '__main__':
    export_fcnet_to_onnx(verbose=True)
