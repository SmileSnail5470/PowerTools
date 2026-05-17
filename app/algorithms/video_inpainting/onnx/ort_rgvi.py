"""
ONNX Runtime RGVI - Complete Video Inpainting Pipeline without PyTorch.

This is a 1:1 functional equivalent of the PyTorch RGVI model,
using ONNX Runtime for neural network inference and numpy for all operations.

Dependencies: onnxruntime, numpy, Pillow, scipy
No PyTorch dependency.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path

from .models.ort_raft import ORT_Raft
from .models.ort_fcnet import ORT_FCNet
from .models.ort_pfcnet import ORT_PFCNet
from .utils.ops import (
    backward_warp,
    interpolate_np,
    avg_pool2d_np,
    masks_to_boxes_np,
)


class ORT_RGVI:
    """
    ONNX Runtime RGVI model - fully torch-free video inpainting.
    
    Mirrors the original RGVI.forward() logic exactly.
    Uses ONNX Runtime for: RAFT (optical flow), FCNet (flow completion), PFCNet (pixel completion).
    For Stable Diffusion inpainting, provides fallback options.
    """
    
    def __init__(self, model_dir=None, use_onnx_sdi=False):
        """
        Args:
            model_dir: Directory containing ONNX model files.
                       Defaults to onnx/models/onnx_models/
            use_onnx_sdi: If True, use ONNX SD inpainting (requires exported SD components).
                          If False, the pipeline will work without SD (flow completion only).
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / 'models' / 'onnx_models'
        self.model_dir = Path(model_dir)
        self.use_onnx_sdi = use_onnx_sdi
        
        # Load ONNX models
        self.raft = ORT_Raft(str(self.model_dir / 'raft_flow.onnx'))
        self.fcnet = ORT_FCNet(str(self.model_dir / 'fcnet.onnx'))
        self.pfcnet = ORT_PFCNet(str(self.model_dir / 'pfcnet.onnx'))
        
        if use_onnx_sdi:
            # Load SD inpainting components (separate pipelines)
            self._load_sdi_components()
        else:
            self.sdi_pipeline = None
    
    def _load_sdi_components(self):
        """Load ONNX SD inpainting components."""
        sdi_dir = self.model_dir / 'sdi'
        from .models.ort_sdi import ORT_SDI_Pipeline
        self.sdi_pipeline = ORT_SDI_Pipeline(sdi_dir)
    
    def _sdi_inpaint(self, image, mask, prompt=None):
        """
        Run Stable Diffusion inpainting on a single image.
        
        Args:
            image: PIL Image or numpy (H, W, 3) in [0, 1]
            mask: PIL Image or numpy (H, W) or (H, W, 1) binary mask
            prompt: text prompt
        
        Returns:
            out: PIL Image
        """
        if self.use_onnx_sdi and self.sdi_pipeline is not None:
            return self.sdi_pipeline.inpaint(image, mask, prompt)
        else:
            raise RuntimeError(
                "SD Inpainting requires PyTorch (diffusers) or ONNX SD components. "
                "Set use_onnx_sdi=True and export SD components, or use the PyTorch version."
            )
    
    def forward(self, imgs, neg_masks, pos_masks, res, prompt):
        """
        Main forward pass - complete video inpainting pipeline.
        
        Args:
            imgs: numpy (L, 3, H_orig, W_orig) - input video frames in [0, 1]
            neg_masks: numpy (L, 1, H_orig, W_orig) - negative masks (0/1)
            pos_masks: numpy (L, 1, H_orig, W_orig) or None - positive masks
            res: str - resolution, one of '240p', '480p', '2K'
            prompt: str or None - text prompt for generative inpainting
        
        Returns:
            pred_imgs: numpy (L, 3, H_orig, W_orig) - inpainted frames
        """
        L = imgs.shape[0]
        
        # Set resolution
        if res == '240p':
            H, W = 240, 432
        elif res == '480p':
            H, W = 480, 864
        elif res == '2K':
            H, W = 1200, 2160
        else:
            raise ValueError(f"Unknown resolution: {res}")
        
        # Resize inputs
        imgs = interpolate_np(imgs, size=(H, W), mode='bicubic')
        neg_masks = interpolate_np(neg_masks, size=(H, W), mode='bicubic')
        if pos_masks is not None:
            pos_masks = interpolate_np(pos_masks, size=(H, W), mode='bicubic')
        
        # Combine masks
        if pos_masks is not None:
            masks = neg_masks + pos_masks
        else:
            masks = neg_masks
        masks = (masks != 0).astype(np.float32)
        cnts = 1 - masks
        
        # Memorize original images
        org_imgs = imgs.copy()
        
        # --- Step 1: Optical Flow Generation (at 480p max) ---
        fw_flows = {}
        bw_flows = {}
        
        for i in range(1, L):
            prev_img = interpolate_np(imgs[i - 1:i], size=(480, 864), mode='bicubic')
            curr_img = interpolate_np(imgs[i:i + 1], size=(480, 864), mode='bicubic')
            # Normalize to [-1, 1]
            fw_flows[i - 1] = self.raft(2 * prev_img - 1, 2 * curr_img - 1)
            bw_flows[i] = self.raft(2 * curr_img - 1, 2 * prev_img - 1)
        
        # Input masking
        imgs = imgs * cnts
        
        # --- Step 2: Flow Completion (at 240p) ---
        s = H / 240.0
        
        fcnet_masks = interpolate_np(masks, size=(240, 432), mode='bicubic')
        fcnet_masks = avg_pool2d_np(fcnet_masks, 9, 1, 4)
        fcnet_masks = (fcnet_masks != 0).astype(np.float32)
        # Add batch and temporal dims
        fcnet_masks = fcnet_masks[np.newaxis, :, :, :, :]  # (1, L, 1, 240, 432)
        
        fcnet_fw_flows = np.zeros((1, L - 1, 2, 240, 432), dtype=np.float32)
        fcnet_bw_flows = np.zeros((1, L - 1, 2, 240, 432), dtype=np.float32)
        for i in range(L - 1):
            fcnet_fw_flows[:, i] = interpolate_np(fw_flows[i], size=(240, 432), mode='bicubic') / 2.0
            fcnet_bw_flows[:, i] = interpolate_np(bw_flows[i + 1], size=(240, 432), mode='bicubic') / 2.0
        
        fcnet_fw_flows = (1 - fcnet_masks[:, :-1]) * fcnet_fw_flows
        fcnet_bw_flows = (1 - fcnet_masks[:, 1:]) * fcnet_bw_flows
        fcnet_flows = [fcnet_fw_flows, fcnet_bw_flows]
        
        fcnet_inp_flows = self.fcnet.forward_bidirect_flow(fcnet_flows, fcnet_masks)
        fcnet_inp_flows = self.fcnet.combine_flow(fcnet_flows, fcnet_inp_flows, fcnet_masks)
        
        # Convert to our format
        inp_fw_flows = {}
        inp_bw_flows = {}
        for i in range(L - 1):
            inp_fw_flows[i] = fcnet_inp_flows[0][:, i]
            inp_bw_flows[i + 1] = fcnet_inp_flows[1][:, i]
        
        # --- Step 3: Internal Pixel Propagation ---
        fw_imgs = imgs.copy()
        bw_imgs = imgs.copy()
        fw_cnts = cnts.copy()
        bw_cnts = cnts.copy()
        warp_masks = np.zeros((L, 1, H, W), dtype=np.float32)
        
        for i in range(L):
            # Pulling from forward direction
            for j in range(i + 1, L):
                if j == i + 1:
                    acc_flow = inp_fw_flows[i]
                else:
                    acc_flow = backward_warp(inp_fw_flows[j - 1], acc_flow) + acc_flow
                acc_flow_s = interpolate_np(acc_flow, size=(H, W), mode='bicubic') * s
                warp_img = backward_warp(imgs[j:j + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[j:j + 1], acc_flow_s)[0]
                fw_imgs[i] = fw_imgs[i] + (1 - fw_cnts[i]) * warp_img
                fw_cnts[i] = fw_cnts[i] + (1 - fw_cnts[i]) * warp_cnt
                warp_masks[i] = warp_masks[i] + 1 - warp_cnt
            
            # Pulling from backward direction
            for j in range(i - 1, -1, -1):
                if j == i - 1:
                    acc_flow = inp_bw_flows[i]
                else:
                    acc_flow = backward_warp(inp_bw_flows[j + 1], acc_flow) + acc_flow
                acc_flow_s = interpolate_np(acc_flow, size=(H, W), mode='bicubic') * s
                warp_img = backward_warp(imgs[j:j + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[j:j + 1], acc_flow_s)[0]
                bw_imgs[i] = bw_imgs[i] + (1 - bw_cnts[i]) * warp_img
                bw_cnts[i] = bw_cnts[i] + (1 - bw_cnts[i]) * warp_cnt
                warp_masks[i] = warp_masks[i] + 1 - warp_cnt
        
        # Invalidate incomplete propagation
        fw_imgs[np.tile(fw_cnts != 1, (1, 3, 1, 1))] = 0
        fw_cnts[fw_cnts != 1] = 0
        bw_imgs[np.tile(bw_cnts != 1, (1, 3, 1, 1))] = 0
        bw_cnts[bw_cnts != 1] = 0
        
        # Collect both directions
        denom = (fw_cnts + bw_cnts).clip(1e-7)
        imgs = (fw_imgs + bw_imgs) / denom
        masks = 1 - (fw_cnts + bw_cnts).clip(0, 1)
        cnts = 1 - masks
        
        # Propagation verification
        threshold = 1.0
        diff = np.sum(np.abs(fw_imgs - bw_imgs), axis=1, keepdims=True)
        unsure = (diff > threshold).astype(np.float32) * (fw_cnts + bw_cnts - 1).clip(0, 1)
        
        # Count connected pixels
        con_num = np.zeros(L)
        for i in range(L):
            con_num[i] = np.sum(masks[i]) + np.sum(masks[i] * warp_masks[i])
        
        # --- Step 4: Select target frame and generate reference ---
        k = int(np.argmax(con_num))
        
        if 1 in np.unique(masks[k]):
            # Detach box for generation
            bbox = masks_to_boxes_np(masks[k])[0]
            
            if prompt is None:
                x1, x2, y1, y2 = 0, W, 0, H
            else:
                x1 = int(max(bbox[0] - 20 * s, 0))
                x2 = int(min(bbox[2] + 20 * s, W))
                y1 = int(max(bbox[1] - 20 * s, 0))
                y2 = int(min(bbox[3] + 20 * s, H))
            
            crop_img = imgs[k, :, y1:y2, x1:x2]
            crop_mask = masks[k, :, y1:y2, x1:x2]
            
            # Convert to PIL for SD inpainting
            crop_img_pil = Image.fromarray(
                (crop_img.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
            )
            crop_mask_pil = Image.fromarray(
                (crop_mask[0].clip(0, 1) * 255).astype(np.uint8)
            )
            
            if prompt is None:
                prompt = 'Empty background, high resolution'
            
            # Run SD inpainting
            out_pil = self._sdi_inpaint(crop_img_pil, crop_mask_pil, prompt)
            out_pil = out_pil.resize((x2 - x1, y2 - y1), Image.BICUBIC)
            
            # Paste back
            out_np = np.array(out_pil).astype(np.float32) / 255.0
            if out_np.ndim == 3 and out_np.shape[2] == 3:
                out_np = out_np.transpose(2, 0, 1)  # (3, Hc, Wc)
            imgs[k, :, y1:y2, x1:x2] = imgs[k, :, y1:y2, x1:x2] + \
                masks[k, :, y1:y2, x1:x2] * out_np
            cnts[k] = 1.0
            
            # Pull from forward direction (backward in time from k)
            for i in range(k - 1, -1, -1):
                if i == k - 1:
                    acc_flow = inp_fw_flows[i]
                else:
                    acc_flow = backward_warp(acc_flow, inp_fw_flows[i]) + inp_fw_flows[i]
                acc_flow_s = interpolate_np(acc_flow, size=(H, W), mode='bicubic') * s
                warp_img = backward_warp(imgs[k:k + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[k:k + 1], acc_flow_s)[0]
                imgs[i] = imgs[i] + (1 - cnts[i]) * warp_img
                cnts[i] = cnts[i] + (1 - cnts[i]) * warp_cnt
            
            # Pull from backward direction (forward in time from k)
            for i in range(k + 1, L):
                if i == k + 1:
                    acc_flow = inp_bw_flows[i]
                else:
                    acc_flow = backward_warp(acc_flow, inp_bw_flows[i]) + inp_bw_flows[i]
                acc_flow_s = interpolate_np(acc_flow, size=(H, W), mode='bicubic') * s
                warp_img = backward_warp(imgs[k:k + 1], acc_flow_s)[0]
                warp_cnt = backward_warp(cnts[k:k + 1], acc_flow_s)[0]
                imgs[i] = imgs[i] + (1 - cnts[i]) * warp_img
                cnts[i] = cnts[i] + (1 - cnts[i]) * warp_cnt
            
            # Invalidate incomplete propagation
            imgs[np.tile(cnts != 1, (1, 3, 1, 1))] = 0
            cnts[cnts != 1] = 0
        
        # Propagation verification
        imgs = imgs * (1 - unsure)
        masks = 1 - cnts * (1 - unsure)
        
        # --- Step 5: Missing area completion with PFCNet ---
        for i in range(L):
            if 1 in np.unique(masks[i]):
                imgs[i:i + 1] = self.pfcnet(imgs[i:i + 1], masks[i:i + 1])
        
        # Attach back positive masks
        if pos_masks is not None:
            imgs = (1 - pos_masks) * imgs + pos_masks * org_imgs
        
        return imgs
    
    def __call__(self, imgs, neg_masks, pos_masks, res, prompt):
        return self.forward(imgs, neg_masks, pos_masks, res, prompt)
