import os
import platform
import sys
import cv2
import numpy as np
import onnxruntime as ort
from app.algorithms import ORTEnvironment, general_session, general_provider
ORTEnvironment.initialize()
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.bidirectional_propagation import BidirectionalPropagationORT, ImgPropStepORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.decoder import DecoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.encoder import EncoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.soft_split_comp import SoftSplitORT, SoftCompORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.transformer_block import TemporalSparseTransformerBlockORT


def interpolate_numpy(x, scale_factor, mode='bilinear'):
    b, c, h, w = x.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    out = np.zeros((b, c, new_h, new_w), dtype=np.float32)
    cv_mode = cv2.INTER_LINEAR if mode == 'bilinear' else cv2.INTER_NEAREST
    for i in range(b):
        img = np.ascontiguousarray(x[i].transpose(1, 2, 0), dtype=np.float32)
        res = cv2.resize(img, (new_w, new_h), interpolation=cv_mode)
        if c == 1:
            out[i, 0] = res
        else:
            out[i] = res.transpose(2, 0, 1)
    return out


def max_pool2d_numpy(x, kernel_size=(7, 7), stride=(3, 3), padding=(3, 3)):
    b, c, h, w = x.shape
    ph, pw = padding
    # 使用极小值进行 padding
    x_pad = np.pad(x, ((0,0), (0,0), (ph, ph), (pw, pw)), mode='constant', constant_values=-1e9)
    out_h = (h + 2 * ph - kernel_size[0]) // stride[0] + 1
    out_w = (w + 2 * pw - kernel_size[1]) // stride[1] + 1
    out = np.zeros((b, c, out_h, out_w), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride[0]
            h_end = h_start + kernel_size[0]
            w_start = j * stride[1]
            w_end = w_start + kernel_size[1]
            out[:, :, i, j] = np.max(x_pad[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
    return out


class PropagationTransformerORT:
    def __init__(self, onnx_paths):
        self.depths = 8
        self.window_size = (5, 9)
        self.onnx_paths = onnx_paths
        self.encoder = None
        self.prepare_models()

    def _get_session_options(self) -> ort.SessionOptions:
        opts = general_session()
        return opts
    
    def __del__(self):
        if hasattr(self, 'encoder'):
            del self.encoder
        if hasattr(self, 'decoder'):
            del self.decoder
        if hasattr(self, 'feat_prop'):
            del self.feat_prop
        if hasattr(self, 'ss'):
            del self.ss
        if hasattr(self, 'sc'):
            del self.sc
        if hasattr(self, 'transformers'):
            del self.transformers
        if hasattr(self, 'img_prop'):
            del self.img_prop
    
    def prepare_models(self):
        if self.encoder is not None:
            return
        sess_options = self._get_session_options()
        providers, provider_options = general_provider()
        run_options = ort.RunOptions()

        self.encoder = EncoderORT(self.onnx_paths['encoder'], providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)
        self.decoder = DecoderORT(self.onnx_paths['decoder'], providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)
        self.feat_prop = BidirectionalPropagationORT(
            backward_step_path=self.onnx_paths['bp_backward_step'],
            forward_step_path=self.onnx_paths['bp_forward_step'],
            backward_first_path=self.onnx_paths['bp_backward_first'],
            forward_first_path=self.onnx_paths['bp_forward_first'],
            fusion_path=self.onnx_paths['bp_fusion'],
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
            run_options=run_options
        )
        self.ss = SoftSplitORT(self.onnx_paths['ss'], providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)
        self.sc = SoftCompORT(self.onnx_paths['sc'], providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)
        self.transformers = TemporalSparseTransformerBlockORT(
            depths=self.depths,
            window_size=self.window_size,
            onnx_paths=self.onnx_paths['transformer'],
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
            run_options=run_options
        )
        self.img_prop = ImgPropStepORT(self.onnx_paths["image_prop_step"], providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)

    def forward(self, masked_frames, completed_flows_f, completed_flows_b, masks_in, masks_updated, num_local_frames, t_dilation=2):
        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.shape

        inp_frames = masked_frames.reshape(b * t, 3, ori_h, ori_w)
        inp_masks_in = masks_in.reshape(b * t, 1, ori_h, ori_w)
        inp_masks_updated = masks_updated.reshape(b * t, 1, ori_h, ori_w)

        enc_feat = self.encoder(inp_frames, inp_masks_in, inp_masks_updated)
        _, c, h, w = enc_feat.shape

        enc_feat = enc_feat.reshape(b, t, c, h, w)
        local_feat = enc_feat[:, :l_t, ...]
        ref_feat = enc_feat[:, l_t:, ...]

        ds_flows_f = interpolate_numpy(completed_flows_f.reshape(-1, 2, ori_h, ori_w), 0.25, mode='bilinear')
        ds_flows_f = ds_flows_f.reshape(b, l_t-1, 2, h, w) / 4.0
        ds_flows_b = interpolate_numpy(completed_flows_b.reshape(-1, 2, ori_h, ori_w), 0.25, mode='bilinear')
        ds_flows_b = ds_flows_b.reshape(b, l_t-1, 2, h, w) / 4.0
        ds_mask_in = interpolate_numpy(masks_in.reshape(-1, 1, ori_h, ori_w), 0.25, mode='nearest')
        ds_mask_in = ds_mask_in.reshape(b, t, 1, h, w)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        masks_updated_local_flat = masks_updated[:, :l_t].reshape(-1, 1, ori_h, ori_w)
        ds_mask_updated_local = interpolate_numpy(masks_updated_local_flat, 0.25, mode='nearest')
        ds_mask_updated_local = ds_mask_updated_local.reshape(b, l_t, 1, h, w)

        mask_pool_l = max_pool2d_numpy(ds_mask_in_local.reshape(-1, 1, h, w), kernel_size=(7,7), stride=(3,3), padding=(3,3))
        mask_pool_l = mask_pool_l.reshape(b, l_t, 1, mask_pool_l.shape[-2], mask_pool_l.shape[-1])

        prop_mask_in = np.concatenate([ds_mask_in_local, ds_mask_updated_local], axis=2)
        local_feat = self.feat_prop.forward(local_feat, ds_flows_f, ds_flows_b, prop_mask_in)
        enc_feat_updated = np.concatenate((local_feat, ref_feat), axis=1)

        enc_feat_flat = enc_feat_updated.reshape(-1, c, h, w)
        trans_feat = self.ss(enc_feat_flat)

        # Rearrange 'b t c h w -> b t h w c' 
        mask_pool_l = np.transpose(mask_pool_l, (0, 1, 3, 4, 2))
        mask_pool_l = np.ascontiguousarray(mask_pool_l)
        trans_feat = self.transformers(trans_feat, enc_feat_flat, mask_pool_l, t_dilation=t_dilation)

        trans_feat = self.sc(trans_feat, enc_feat_flat)
        trans_feat = trans_feat.reshape(b, t, -1, h, w)

        enc_feat_updated = enc_feat_updated + trans_feat

        output = self.decoder(enc_feat_updated[:, :l_t].reshape(-1, c, h, w))
        output = np.tanh(output).reshape(b, l_t, 3, ori_h, ori_w)
        return output
    
    def img_propagation(self, masked_frames, completed_flows, masks):
        flows_forward, flows_backward = completed_flows
        b, t, c, h, w = masked_frames.shape
        feats_input = [masked_frames[:, i, :, :, :] for i in range(t)]
        masks_input = [masks[:, i, :, :, :] for i in range(t)]

        feats_b = []
        masks_b = []
        frame_idx_b = list(range(0, t))[::-1]
        flow_idx_b = frame_idx_b
        for i, idx in enumerate(frame_idx_b):
            feat_current = feats_input[idx]
            mask_current = masks_input[idx]
            if i == 0:
                feat_prop = feat_current
                mask_prop = mask_current
            else:
                flow_prop = flows_forward[:, flow_idx_b[i], :, :, :]
                flow_check = flows_backward[:, flow_idx_b[i], :, :, :]
                feat_prop, mask_prop = self.img_prop(
                    feat_current, feat_prop, mask_current, mask_prop, flow_prop, flow_check
                )
            feats_b.append(feat_prop)
            masks_b.append(mask_prop)
        feats_b = feats_b[::-1]
        masks_b = masks_b[::-1]

        feats_f = []
        masks_f = []
        frame_idx_f = list(range(0, t))
        flow_idx_f = list(range(-1, t - 1))
        for i, idx in enumerate(frame_idx_f):
            feat_current = feats_b[idx]
            mask_current = masks_b[idx]
            if i == 0:
                feat_prop = feat_current
                mask_prop = mask_current
            else:
                flow_prop = flows_backward[:, flow_idx_f[i], :, :, :]
                flow_check = flows_forward[:, flow_idx_f[i], :, :, :]
                
                feat_prop, mask_prop = self.img_prop(
                    feat_current, feat_prop, mask_current, mask_prop, flow_prop, flow_check
                )
            feats_f.append(feat_prop)
            masks_f.append(mask_prop)

        prop_frames = np.stack(feats_f, axis=1).reshape(b, t, c, h, w)
        updated_masks = np.stack(masks_f, axis=1)
        return prop_frames, updated_masks