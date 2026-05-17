"""
ONNX-based Video Inpainting Inference Script.
Completely torch-free inference using ONNX Runtime.

Usage:
    python -m onnx.run_onnx --root input/ --res 480p
    python -m onnx.run_onnx --root input/ --res 480p --prompt "Empty background"
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from .ort_rgvi import ORT_RGVI
from .utils.ops import interpolate_np


class DatasetONNX:
    """Dataset reader - no PyTorch dependency."""
    
    def __init__(self, root):
        self.root = root
    
    def read_img(self, path):
        pic = Image.open(path).convert('RGB')
        img_np = np.array(pic).astype(np.float32) / 255.0
        return img_np.transpose(2, 0, 1)  # (3, H, W)
    
    def read_mask(self, path):
        pic = Image.open(path).convert('L')
        mask_np = np.array(pic).astype(np.float32) / 255.0
        return mask_np[None, :, :]  # (1, H, W)
    
    def get_video(self):
        frames_dir = os.path.join(self.root, 'frames')
        masks_dir = os.path.join(self.root, 'masks')
        pos_mask_dir = os.path.join(self.root, 'PosAnnotations')
        
        frame_ids = sorted([
            int(os.path.splitext(f)[0]) for f in os.listdir(frames_dir)
        ])
        
        imgs = []
        neg_masks = []
        pos_masks = []
        files = []
        
        for fid in frame_ids:
            img_path = os.path.join(frames_dir, f'{fid:06d}.png')
            neg_mask_path = os.path.join(masks_dir, f'{fid:06d}.png')
            pos_mask_path = os.path.join(pos_mask_dir, f'{fid:06d}.png')
            
            imgs.append(self.read_img(img_path))
            neg_masks.append(self.read_mask(neg_mask_path))
            
            if os.path.exists(pos_mask_path):
                pos_masks.append(self.read_mask(pos_mask_path))
            
            files.append(f'{fid:06d}.jpg')
        
        imgs = np.stack(imgs, axis=0)
        neg_masks = np.stack(neg_masks, axis=0)
        
        if len(pos_masks) > 0:
            pos_masks = np.stack(pos_masks, axis=0)
        else:
            pos_masks = None
        
        return {'imgs': imgs, 'neg_masks': neg_masks, 'pos_masks': pos_masks, 'files': files}


class EvaluatorONNX:
    """Evaluator - no PyTorch dependency."""
    
    def __init__(self, root, res):
        self.res = res
        self.dataset = DatasetONNX(root)
    
    def save_image(self, img_np, fpath):
        """Save numpy image (C, H, W) in [0, 1] to file."""
        img_np = (img_np.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(fpath)
    
    def evaluate(self, model, prompt, output_path):
        vi_data = self.dataset.get_video()
        os.makedirs(output_path, exist_ok=True)
        
        imgs = vi_data['imgs']
        neg_masks = vi_data['neg_masks']
        pos_masks = vi_data['pos_masks']
        files = vi_data['files']
        
        # Inference
        t0 = time.time()
        pred_imgs = model(imgs, neg_masks, pos_masks, self.res, prompt)
        t1 = time.time()
        
        # Save output
        L = len(files)
        for i in range(L):
            fpath = os.path.join(output_path, files[i])
            self.save_image(pred_imgs[i], fpath)
        
        seconds = t1 - t0
        print(f'{seconds:.1f} seconds, {L} frames, {L / seconds:.1f} FPS')
        return seconds, L


def main():
    parser = ArgumentParser()
    parser.add_argument('--root', default='input/', type=str, help='root directory of videos')
    parser.add_argument('--res', default='480p', choices=['240p', '480p', '2K'], help='input resolution')
    parser.add_argument('--prompt', default=None, type=str, help='text prompt for generative model')
    parser.add_argument('--model_dir', default=None, type=str, help='path to ONNX model directory')
    parser.add_argument('--use_onnx_sdi', action='store_true', help='use ONNX SD inpainting (experimental)')
    args = parser.parse_args()
    
    print("Initializing ONNX RGVI model...")
    model = ORT_RGVI(model_dir=args.model_dir, use_onnx_sdi=args.use_onnx_sdi)
    
    evaluator = EvaluatorONNX(args.root, args.res)
    evaluator.evaluate(model, args.prompt, "output_onnx")


if __name__ == '__main__':
    main()
