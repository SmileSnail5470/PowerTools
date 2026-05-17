"""
Export Stable Diffusion Inpainting components (UNet, VAE, text_encoder) to ONNX.

Since Stable Diffusion is a large model, we export individual components separately.
The UNet is the main component that does the heavy lifting.

Usage:
    python -m onnx.convert.export_sdi
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
from pathlib import Path


def export_sdi_components(output_dir=None, verbose=False):
    """
    Export Stable Diffusion Inpainting pipeline components to ONNX.
    
    Exports:
        - text_encoder: CLIP text encoder
        - unet: Inpainting UNet
        - vae_encoder: VAE encoder
        - vae_decoder: VAE decoder
    
    Args:
        output_dir: Output directory for ONNX models
        verbose: Print detailed logs
    """
    if output_dir is None:
        base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        output_dir = base_dir / 'models' / 'onnx_models' / 'sdi'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cpu'

    from diffusers import StableDiffusionInpaintPipeline as SDI
    from diffusers import DPMSolverMultistepScheduler
    
    weights_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'weights', 'stable-diffusion-2-inpainting'
    )
    
    if verbose:
        print(f"Loading SD inpainting pipeline from {weights_dir}...")
    
    pipe = SDI.from_pretrained(
        weights_dir,
        torch_dtype=torch.float32,
        cache_dir='../huggingface'
    ).to(device)
    
    # Use a simpler scheduler for deterministic ONNX export
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # --- Export text encoder ---
    if verbose:
        print("Exporting text encoder...")
    
    text_encoder = pipe.text_encoder.eval()
    dummy_input_ids = torch.randint(0, 49407, (1, 77), device=device)
    
    text_encoder_path = output_dir / 'text_encoder.onnx'
    torch.onnx.export(
        text_encoder, dummy_input_ids, str(text_encoder_path),
        input_names=['input_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'},
        },
        opset_version=17,
        verbose=verbose,
    )
    if verbose:
        print(f"  -> {text_encoder_path} ({text_encoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # --- Export UNet ---
    if verbose:
        print("Exporting UNet (inpainting)...")
    
    unet = pipe.unet.eval()
    
    # UNet takes: sample, timestep, encoder_hidden_states, (and for inpainting: concat_conds)
    # Inpainting UNet has 9 input channels: 4 (noisy latent) + 4 (masked image latent) + 1 (mask)
    dummy_sample = torch.randn(1, 4, 64, 64, device=device)
    dummy_timestep = torch.tensor([999], device=device)
    dummy_encoder_hidden = torch.randn(1, 77, 1024, device=device)
    
    unet_path = output_dir / 'unet.onnx'
    torch.onnx.export(
        unet,
        (dummy_sample, dummy_timestep, dummy_encoder_hidden),
        str(unet_path),
        input_names=['sample', 'timestep', 'encoder_hidden_states'],
        output_names=['out_sample'],
        dynamic_axes={
            'sample': {0: 'batch'},
            'encoder_hidden_states': {0: 'batch'},
            'out_sample': {0: 'batch'},
        },
        opset_version=17,
        verbose=verbose,
    )
    if verbose:
        print(f"  -> {unet_path} ({unet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # --- Export VAE encoder ---
    if verbose:
        print("Exporting VAE encoder...")
    
    vae = pipe.vae.eval()
    dummy_vae_input = torch.randn(1, 3, 512, 512, device=device)
    
    vae_encoder_path = output_dir / 'vae_encoder.onnx'
    
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, x):
            return self.vae.encode(x).latent_dist.sample()
    
    vae_encoder = VAEEncoderWrapper(vae).eval()
    
    torch.onnx.export(
        vae_encoder, dummy_vae_input, str(vae_encoder_path),
        input_names=['pixel_values'],
        output_names=['latent'],
        dynamic_axes={
            'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
            'latent': {0: 'batch'},
        },
        opset_version=17,
        verbose=verbose,
    )
    if verbose:
        print(f"  -> {vae_encoder_path} ({vae_encoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # --- Export VAE decoder ---
    if verbose:
        print("Exporting VAE decoder...")
    
    dummy_latent = torch.randn(1, 4, 64, 64, device=device)
    vae_decoder_path = output_dir / 'vae_decoder.onnx'
    
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, z):
            return self.vae.decode(z).sample
    
    vae_decoder = VAEDecoderWrapper(vae).eval()
    
    torch.onnx.export(
        vae_decoder, dummy_latent, str(vae_decoder_path),
        input_names=['latent'],
        output_names=['pixel_values'],
        dynamic_axes={
            'latent': {0: 'batch'},
            'pixel_values': {0: 'batch'},
        },
        opset_version=17,
        verbose=verbose,
    )
    if verbose:
        print(f"  -> {vae_decoder_path} ({vae_decoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    if verbose:
        print("\nAll SDI components exported successfully!")
        print(f"\nNote: To use these ONNX models, you also need the tokenizer and scheduler configs")
        print(f"from the original SD weights directory: {weights_dir}")
    
    return str(output_dir)


if __name__ == '__main__':
    export_sdi_components(verbose=True)
