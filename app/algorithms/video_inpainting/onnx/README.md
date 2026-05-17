# ONNX Runtime Video Inpainting (RGVI)

Torch-free video inpainting inference using ONNX Runtime.
Functional 1:1 equivalent of the original PyTorch RGVI model.

## Architecture

The ONNX conversion exports 3 neural network components:
- **RAFT**: Optical flow estimation (from torchvision)
- **FCNet**: Flow completion network (custom mmcv ops replaced with grid_sample decomposition)
- **PFCNet**: Pixel-level flow completion network (pure PyTorch, direct export)

Additionally, Stable Diffusion Inpainting (SDI) components can be exported separately.

## Directory Structure

```
onnx/
├── convert/                    # Export scripts (requires PyTorch)
│   ├── export_raft.py          # Export RAFT to ONNX
│   ├── export_fcnet.py         # Export FCNet to ONNX (patches mmcv ops)
│   ├── export_pfcnet.py        # Export PFCNet to ONNX
│   ├── export_sdi.py           # Export SD inpainting components
│   └── export_all.py           # Master export script
├── models/                     # ONNX Runtime inference wrappers
│   ├── ort_raft.py             # RAFT ONNX inference (no torch)
│   ├── ort_fcnet.py            # FCNet ONNX inference (no torch)
│   ├── ort_pfcnet.py           # PFCNet ONNX inference (no torch)
│   ├── ort_sdi.py              # SD inpainting ONNX pipeline
│   ├── onnx_models/            # Exported .onnx files (generated)
│   │   ├── raft_flow.onnx
│   │   ├── fcnet.onnx
│   │   ├── pfcnet.onnx
│   │   └── sdi/                # SD components (optional)
│   └── __init__.py
├── utils/
│   ├── ops.py                  # Numpy ops: grid_sample, warp, resize, etc.
│   └── __init__.py
├── ort_rgvi.py                 # Main ORT_RGVI model (torch-free)
├── run_onnx.py                 # Inference entry point
├── requirements_onnx.txt       # Runtime dependencies (torch-free)
└── README.md
```

## Workflow

### Step 1: Export ONNX models (on a machine with PyTorch)

```bash
cd video_inpainting

# Export all models
python -m onnx.convert.export_all

# Or export individual models
python -m onnx.convert.export_raft
python -m onnx.convert.export_fcnet
python -m onnx.convert.export_pfcnet

# Optional: export SD inpainting components
python -m onnx.convert.export_all --export_sdi
```

### Step 2: Run inference (torch-free)

```bash
# Install runtime dependencies
pip install -r onnx/requirements_onnx.txt

# Run inference
python -m onnx.run_onnx --root input/ --res 480p
python -m onnx.run_onnx --root input/ --res 480p --prompt "Empty background"
```

## Key Technical Details

### FCNet Export (ModulatedDeformConv2d)

FCNet uses `ModulatedDeformConv2d` from mmcv, which is a custom CUDA operator
not directly exportable to ONNX. The export script:

1. Replaces `ModulatedDeformConv2d` with a pure-PyTorch equivalent that
   decomposes the deformable convolution into per-kernel-element grid_sample
   operations
2. This produces an ONNX graph using only standard operators (Conv, GridSample,
   etc.) supported by ONNX Runtime
3. The decomposition is mathematically equivalent to the original CUDA operation

### Numerical Equivalence

The ONNX pipeline uses:
- ONNX Runtime for neural network inference (RAFT, FCNet, PFCNet)
- Numpy implementations of grid_sample, bicubic interpolation, and other ops
- The same algorithms and data flow as the original PyTorch version

Output differences compared to PyTorch are limited to:
- Floating point precision differences (ONNX Runtime vs PyTorch)
- Interpolation differences (numpy bicubic vs PyTorch bicubic)

## Dependencies

### Runtime (torch-free):
- onnxruntime>=1.15.0
- numpy>=1.22.0
- Pillow>=9.0.0
- scipy>=1.7.0

### Export (requires PyTorch):
- torch>=2.0.0
- torchvision>=0.15.0
- mmcv>=2.0.0
- diffusers>=0.21.0 (for SD export)
- transformers>=4.30.0 (for SD export)
