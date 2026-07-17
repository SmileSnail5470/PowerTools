import cv2
import numpy as np
import onnxruntime as ort
from app.algorithms import ORTEnvironment, general_provider, ortvalue_to_numpy
from app.algorithms.visible_watermark_removal.video_modules.ppt.runtime import ppt_run_options, ppt_session_options
ORTEnvironment.initialize()
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.bidirectional_propagation import BidirectionalPropagationORT, ImgPropStepORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.decoder import DecoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.encoder import EncoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.soft_split_comp import SoftSplitORT, SoftCompORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.transformer_block import TemporalSparseTransformerBlockORT, _ortvalue_to_cupy, _cupy_to_ortvalue
try:
    import cupy as cp
    import cupyx.scipy.ndimage
except ImportError:
    pass


def interpolate_numpy(x, scale_factor, mode='bilinear'):
    b, c, h, w = x.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    cv_mode = cv2.INTER_LINEAR if mode == 'bilinear' else cv2.INTER_NEAREST
    if c <= 4:
        out = np.zeros((b, c, new_h, new_w), dtype=np.float32)
        for i in range(b):
            img = np.ascontiguousarray(x[i].transpose(1, 2, 0), dtype=np.float32)
            res = cv2.resize(img, (new_w, new_h), interpolation=cv_mode)
            if c == 1:
                out[i, 0] = res if res.ndim == 2 else res[:, :, 0]
            else:
                out[i] = res.transpose(2, 0, 1)
    else:
        out = np.zeros((b, c, new_h, new_w), dtype=np.float32)
        for i in range(b):
            for ch in range(c):
                img = np.ascontiguousarray(x[i, ch], dtype=np.float32)
                out[i, ch] = cv2.resize(img, (new_w, new_h), interpolation=cv_mode)
    return out


def interpolate_cupy(x, scale_factor, mode='bilinear'):
    b, c, h, w = x.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    if mode == 'nearest':
        order = 0
    else:
        order = 1
    zoom_h = new_h / h
    zoom_w = new_w / w
    return cupyx.scipy.ndimage.zoom(x, (1.0, 1.0, zoom_h, zoom_w), order=order)


def max_pool2d_numpy(x, kernel_size=(7, 7), stride=(3, 3), padding=(3, 3)):
    b, c, h, w = x.shape
    ph, pw = padding
    kh, kw = kernel_size
    sh, sw = stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=-1e9)
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    s_b, s_c, s_h, s_w = x_pad.strides
    shape = (b, c, out_h, out_w, kh, kw)
    strides = (s_b, s_c, s_h * sh, s_w * sw, s_h, s_w)
    windows = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)
    return np.ascontiguousarray(windows.max(axis=(4, 5)))


def max_pool2d_cupy(x, kernel_size=(7, 7), stride=(3, 3), padding=(3, 3)):
    b, c, h, w = x.shape
    ph, pw = padding
    kh, kw = kernel_size
    sh, sw = stride
    x_pad = cp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=-1e9)
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    out = cp.empty((b, c, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = x_pad[:, :, i:i + out_h * sh:sh, j:j + out_w * sw:sw]
            if i == 0 and j == 0:
                out[:] = patch
            else:
                cp.maximum(out, patch, out=out)
    return out


class PropagationTransformerORT:
    def __init__(self, onnx_paths):
        self.depths = 8
        self.window_size = (5, 9)
        self.onnx_paths = onnx_paths
        self.encoder = None
        self.prepare_models()

    def _get_session_options(self) -> ort.SessionOptions:
        return ppt_session_options()

    def __del__(self):
        for attr in ('encoder', 'decoder', 'feat_prop', 'ss', 'sc', 'transformers', 'img_prop'):
            if hasattr(self, attr):
                delattr(self, attr)

    def prepare_models(self):
        if self.encoder is not None:
            return
        sess_options = self._get_session_options()
        providers, provider_options = general_provider()
        run_options = ppt_run_options()

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
        self._use_cupy = self.encoder._use_cupy
        self.shrink_run_options = ppt_run_options(shrink_memory=True, use_cuda=self.encoder._use_iobinding)

    def forward(
        self,
        masked_frames,
        completed_flows_f,
        completed_flows_b,
        masks_in,
        masks_updated,
        num_local_frames,
        t_dilation=2,
        shrink_memory=False,
    ):
        if self._use_cupy:
            return self._forward_cupy(
                masked_frames,
                completed_flows_f,
                completed_flows_b,
                masks_in,
                masks_updated,
                num_local_frames,
                t_dilation,
                shrink_memory,
            )
        return self._forward_cpu(
            masked_frames,
            completed_flows_f,
            completed_flows_b,
            masks_in,
            masks_updated,
            num_local_frames,
            t_dilation,
            shrink_memory,
        )

    def _forward_cpu(
        self,
        masked_frames,
        completed_flows_f,
        completed_flows_b,
        masks_in,
        masks_updated,
        num_local_frames,
        t_dilation=2,
        shrink_memory=False,
    ):
        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.shape
        masks_in = np.asarray(masks_in, dtype=np.float32)
        masks_updated = np.asarray(masks_updated, dtype=np.float32)

        enc_feat = self.encoder(masked_frames.reshape(b * t, 3, ori_h, ori_w),
                                masks_in.reshape(b * t, 1, ori_h, ori_w),
                                masks_updated.reshape(b * t, 1, ori_h, ori_w))
        _, c, h, w = enc_feat.shape
        enc_feat = enc_feat.reshape(b, t, c, h, w)
        local_feat = enc_feat[:, :l_t, ...]
        ref_feat = enc_feat[:, l_t:, ...]

        ds_flows_f = interpolate_numpy(completed_flows_f.reshape(-1, 2, ori_h, ori_w), 0.25, mode='bilinear').reshape(b, l_t - 1, 2, h, w) / 4.0
        ds_flows_b = interpolate_numpy(completed_flows_b.reshape(-1, 2, ori_h, ori_w), 0.25, mode='bilinear').reshape(b, l_t - 1, 2, h, w) / 4.0
        ds_mask_in = interpolate_numpy(masks_in.reshape(-1, 1, ori_h, ori_w), 0.25, mode='nearest').reshape(b, t, 1, h, w)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = interpolate_numpy(masks_updated[:, :l_t].reshape(-1, 1, ori_h, ori_w), 0.25, mode='nearest').reshape(b, l_t, 1, h, w)

        mask_pool_l = max_pool2d_numpy(ds_mask_in_local.reshape(-1, 1, h, w), kernel_size=(7, 7), stride=(3, 3), padding=(3, 3))
        mask_pool_l = mask_pool_l.reshape(b, l_t, 1, mask_pool_l.shape[-2], mask_pool_l.shape[-1])

        prop_mask_in = np.concatenate([ds_mask_in_local, ds_mask_updated_local], axis=2)
        local_feat = self.feat_prop.forward(local_feat, ds_flows_f, ds_flows_b, prop_mask_in)
        enc_feat_updated = np.concatenate((local_feat, ref_feat), axis=1)

        enc_feat_flat = enc_feat_updated.reshape(-1, c, h, w)
        trans_feat = self.ss(enc_feat_flat)
        mask_pool_l = np.ascontiguousarray(np.transpose(mask_pool_l, (0, 1, 3, 4, 2)))
        trans_feat = self.transformers(trans_feat, enc_feat_flat, mask_pool_l, t_dilation=t_dilation)
        trans_feat = self.sc(trans_feat, enc_feat_flat)
        trans_feat = trans_feat.reshape(b, t, -1, h, w)
        enc_feat_updated = enc_feat_updated + trans_feat

        output = self.decoder(
            enc_feat_updated[:, :l_t].reshape(-1, c, h, w),
            run_options=(self.shrink_run_options if shrink_memory else None),
        )
        return np.tanh(output).reshape(b, l_t, 3, ori_h, ori_w)

    def _forward_cupy(
        self,
        masked_frames,
        completed_flows_f,
        completed_flows_b,
        masks_in,
        masks_updated,
        num_local_frames,
        t_dilation=2,
        shrink_memory=False,
    ):
        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.shape

        mf = cp.asarray(masked_frames)
        mi = cp.asarray(masks_in, dtype=cp.float32)
        mu = cp.asarray(masks_updated, dtype=cp.float32)

        enc_feat = self.encoder(mf.reshape(b * t, 3, ori_h, ori_w),
                                mi.reshape(b * t, 1, ori_h, ori_w),
                                mu.reshape(b * t, 1, ori_h, ori_w))
        _, c, h, w = enc_feat.shape
        enc_feat = enc_feat.reshape(b, t, c, h, w)
        local_feat = enc_feat[:, :l_t, ...]
        ref_feat = enc_feat[:, l_t:, ...]

        flows_f_flat = cp.asarray(completed_flows_f).reshape(-1, 2, ori_h, ori_w)
        flows_b_flat = cp.asarray(completed_flows_b).reshape(-1, 2, ori_h, ori_w)
        ds_flows_f = interpolate_cupy(flows_f_flat, 0.25, mode='bilinear').reshape(b, l_t - 1, 2, h, w) / 4.0
        ds_flows_b = interpolate_cupy(flows_b_flat, 0.25, mode='bilinear').reshape(b, l_t - 1, 2, h, w) / 4.0
        ds_mask_in = interpolate_cupy(mi.reshape(-1, 1, ori_h, ori_w), 0.25, mode='nearest').reshape(b, t, 1, h, w)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = interpolate_cupy(mu[:, :l_t].reshape(-1, 1, ori_h, ori_w), 0.25, mode='nearest').reshape(b, l_t, 1, h, w)
        del mf, mi, mu

        mask_pool_l = max_pool2d_cupy(ds_mask_in_local.reshape(-1, 1, h, w), kernel_size=(7, 7), stride=(3, 3), padding=(3, 3))
        mask_pool_l = mask_pool_l.reshape(b, l_t, 1, mask_pool_l.shape[-2], mask_pool_l.shape[-1])

        prop_mask_in = cp.concatenate([ds_mask_in_local, ds_mask_updated_local], axis=2)
        local_feat = self.feat_prop.forward(local_feat, ds_flows_f, ds_flows_b, prop_mask_in)
        del ds_flows_f, ds_flows_b, prop_mask_in

        enc_feat_updated = cp.concatenate((local_feat, ref_feat), axis=1)
        del local_feat, ref_feat, enc_feat

        enc_feat_flat = enc_feat_updated.reshape(-1, c, h, w)
        trans_feat = self.ss(enc_feat_flat)
        # mask_pool_l: (b, l_t, 1, ph, pw) → (b, l_t, ph, pw, 1)
        mask_pool_l = cp.ascontiguousarray(mask_pool_l.transpose(0, 1, 3, 4, 2))
        trans_feat = self.transformers(trans_feat, enc_feat_flat, mask_pool_l, t_dilation=t_dilation)
        trans_feat = self.sc(trans_feat, enc_feat_flat)
        del enc_feat_flat
        trans_feat = trans_feat.reshape(b, t, -1, h, w)
        enc_feat_updated = enc_feat_updated + trans_feat
        del trans_feat

        output = self.decoder(
            enc_feat_updated[:, :l_t].reshape(-1, c, h, w),
            run_options=(self.shrink_run_options if shrink_memory else None),
        )
        del enc_feat_updated
        output = cp.tanh(output).reshape(b, l_t, 3, ori_h, ori_w)
        return output

    def img_propagation(self, masked_frames, completed_flows, masks, shrink_memory=False):
        if self._use_cupy:
            return self._img_propagation_cupy(masked_frames, completed_flows, masks, shrink_memory)
        return self._img_propagation_cpu(masked_frames, completed_flows, masks, shrink_memory)

    def _img_propagation_cpu(self, masked_frames, completed_flows, masks, shrink_memory=False):
        flows_forward, flows_backward = completed_flows
        masks = np.asarray(masks, dtype=np.float32)
        b, t, c, h, w = masked_frames.shape
        feats_input = [np.ascontiguousarray(masked_frames[:, i]) for i in range(t)]
        masks_input = [np.ascontiguousarray(masks[:, i]) for i in range(t)]
        use_iobinding = self.img_prop._use_iobinding

        # Backward pass
        feats_b = [None] * t
        masks_b = [None] * t
        feat_prop = None
        mask_prop = None
        for i, idx in enumerate(range(t - 1, -1, -1)):
            if i == 0:
                if use_iobinding:
                    feat_prop = ort.OrtValue.ortvalue_from_numpy(feats_input[idx], device_type="cuda", device_id=0)
                    mask_prop = ort.OrtValue.ortvalue_from_numpy(masks_input[idx], device_type="cuda", device_id=0)
                else:
                    feat_prop = feats_input[idx]
                    mask_prop = masks_input[idx]
            else:
                flow_prop = np.ascontiguousarray(flows_forward[:, idx])
                flow_check = np.ascontiguousarray(flows_backward[:, idx])
                feat_prop, mask_prop = self.img_prop(feats_input[idx], feat_prop, masks_input[idx], mask_prop, flow_prop, flow_check)
            feats_b[idx] = ortvalue_to_numpy(feat_prop)
            masks_b[idx] = ortvalue_to_numpy(mask_prop)

        # Forward pass
        feats_f = [None] * t
        masks_f = [None] * t
        feat_prop = None
        mask_prop = None
        for i, idx in enumerate(range(t)):
            if i == 0:
                if use_iobinding:
                    feat_prop = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(feats_b[idx]), device_type="cuda", device_id=0)
                    mask_prop = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(masks_b[idx]), device_type="cuda", device_id=0)
                else:
                    feat_prop = feats_b[idx]
                    mask_prop = masks_b[idx]
            else:
                flow_prop = np.ascontiguousarray(flows_backward[:, i - 1])
                flow_check = np.ascontiguousarray(flows_forward[:, i - 1])
                feat_prop, mask_prop = self.img_prop(
                    feats_b[idx],
                    feat_prop,
                    masks_b[idx],
                    mask_prop,
                    flow_prop,
                    flow_check,
                    run_options=(
                        self.shrink_run_options
                        if shrink_memory and idx == t - 1
                        else None
                    ),
                )
            feats_f[idx] = ortvalue_to_numpy(feat_prop)
            masks_f[idx] = ortvalue_to_numpy(mask_prop)

        return np.stack(feats_f, axis=1).reshape(b, t, c, h, w), np.stack(masks_f, axis=1)

    def _img_propagation_cupy(self, masked_frames, completed_flows, masks, shrink_memory=False):
        flows_forward, flows_backward = completed_flows
        b, t, c, h, w = masked_frames.shape
        mf_cp = cp.asarray(masked_frames)
        masks_cp = cp.asarray(masks, dtype=cp.float32)
        flows_f_cp = cp.asarray(flows_forward)
        flows_b_cp = cp.asarray(flows_backward)

        # Backward pass — feats stay as OrtValue on GPU
        feats_b_ort = [None] * t
        masks_b_ort = [None] * t
        for i, idx in enumerate(range(t - 1, -1, -1)):
            feat_current = cp.ascontiguousarray(mf_cp[:, idx])
            mask_current = cp.ascontiguousarray(masks_cp[:, idx])
            if i == 0:
                feat_prop = _cupy_to_ortvalue(feat_current)
                mask_prop = _cupy_to_ortvalue(mask_current)
            else:
                flow_prop = cp.ascontiguousarray(flows_f_cp[:, idx])
                flow_check = cp.ascontiguousarray(flows_b_cp[:, idx])
                feat_prop, mask_prop = self.img_prop(feat_current, feat_prop, mask_current, mask_prop, flow_prop, flow_check)
            feats_b_ort[idx] = feat_prop
            masks_b_ort[idx] = mask_prop

        # Forward pass
        feats_f_ort = [None] * t
        masks_f_ort = [None] * t
        for i, idx in enumerate(range(t)):
            feat_current = _ortvalue_to_cupy(feats_b_ort[idx])
            mask_current = _ortvalue_to_cupy(masks_b_ort[idx])
            if i == 0:
                feat_prop = _cupy_to_ortvalue(cp.ascontiguousarray(feat_current))
                mask_prop = _cupy_to_ortvalue(cp.ascontiguousarray(mask_current))
            else:
                flow_prop = cp.ascontiguousarray(flows_b_cp[:, i - 1])
                flow_check = cp.ascontiguousarray(flows_f_cp[:, i - 1])
                feat_prop, mask_prop = self.img_prop(
                    feat_current,
                    feat_prop,
                    mask_current,
                    mask_prop,
                    flow_prop,
                    flow_check,
                    run_options=(
                        self.shrink_run_options
                        if shrink_memory and idx == t - 1
                        else None
                    ),
                )
            feats_f_ort[idx] = feat_prop
            masks_f_ort[idx] = mask_prop

        # Download results
        feats_f = cp.stack([_ortvalue_to_cupy(f) for f in feats_f_ort], axis=1).reshape(b, t, c, h, w)
        masks_f = cp.stack([_ortvalue_to_cupy(m) for m in masks_f_ort], axis=1)
        return cp.asnumpy(feats_f), cp.asnumpy(masks_f)
