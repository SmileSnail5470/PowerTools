"""
Export all models to ONNX format.
Run this script once to generate all ONNX models needed for inference.

Usage:
    python -m onnx.convert.export_all [--output_dir PATH] [--export_sdi]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from pathlib import Path


def export_all(output_dir=None, export_sdi=False, verbose=True):
    """
    Export all models to ONNX format.
    
    Args:
        output_dir: Directory to save ONNX models
        export_sdi: Also export Stable Diffusion Inpainting components (requires diffusers)
        verbose: Print detailed logs
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'models' / 'onnx_models'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Exporting all models to ONNX format")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # 1. Export RAFT
    print("\n[1/3] Exporting RAFT optical flow model...")
    from onnx.convert.export_raft import export_raft_to_onnx
    raft_path = export_raft_to_onnx(str(output_dir), verbose=verbose)
    print(f"  ✓ RAFT exported to {raft_path}")
    
    # 2. Export FCNet
    print("\n[2/3] Exporting FCNet flow completion model...")
    from onnx.convert.export_fcnet import export_fcnet_to_onnx
    fcnet_path = export_fcnet_to_onnx(str(output_dir), verbose=verbose)
    print(f"  ✓ FCNet exported to {fcnet_path}")
    
    # 3. Export PFCNet
    print("\n[3/3] Exporting PFCNet pixel completion model...")
    from onnx.convert.export_pfcnet import export_pfcnet_to_onnx
    pfcnet_path = export_pfcnet_to_onnx(str(output_dir), verbose=verbose)
    print(f"  ✓ PFCNet exported to {pfcnet_path}")
    
    # 4. Export SDI (optional)
    if export_sdi:
        print("\n[4/4] Exporting SD Inpainting components (this may take a while)...")
        from onnx.convert.export_sdi import export_sdi_components
        sdi_path = export_sdi_components(str(output_dir), verbose=verbose)
        print(f"  ✓ SDI components exported to {sdi_path}")
    
    print("\n" + "=" * 60)
    print("All models exported successfully!")
    print(f"ONNX models saved to: {output_dir}")
    print("=" * 60)
    
    # Summary
    print("\nExported files:")
    for f in sorted(output_dir.iterdir()):
        if f.suffix == '.onnx':
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
    
    if (output_dir / 'sdi').exists():
        print("\nSDI components:")
        for f in sorted((output_dir / 'sdi').iterdir()):
            if f.suffix == '.onnx':
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  sdi/{f.name}: {size_mb:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export all models to ONNX')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for ONNX models')
    parser.add_argument('--export_sdi', action='store_true',
                        help='Also export SD Inpainting components')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed logs')
    args = parser.parse_args()
    
    export_all(output_dir=args.output_dir, export_sdi=args.export_sdi, verbose=not args.quiet)
