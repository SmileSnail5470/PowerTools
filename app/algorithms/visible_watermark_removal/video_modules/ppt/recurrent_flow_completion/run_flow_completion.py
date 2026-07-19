import os
import numpy as np
from app.algorithms import ORTEnvironment, general_provider
from app.algorithms.visible_watermark_removal.video_modules.ppt.runtime import ppt_run_options, ppt_session_options
ORTEnvironment.initialize()
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.encoder import EncoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.decoder import DecoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.bidirectional_propagation import BidirectionalPropagationORT
try:
    import cupy as cp
except ImportError:
    pass


class RecurrentFlowCompleteORT:
    def __init__(self, onnx_dir):
        self.onnx_dir = onnx_dir
        self.encoder_ort = None
        self.prop_ort = None
        self.decoder_ort = None
        self.prepare_models()

    def _get_session_options(self):
        return ppt_session_options()

    def __del__(self):
        for attr in ('encoder_ort', 'prop_ort', 'decoder_ort'):
            if hasattr(self, attr):
                delattr(self, attr)

    def prepare_models(self):
        if self.encoder_ort is not None:
            return
        sess_options = self._get_session_options()
        providers, provider_options = general_provider()
        run_options = ppt_run_options()
        self.encoder_ort = EncoderORT(os.path.join(self.onnx_dir, 'encoder.encmodel'), providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)
        self.prop_ort = BidirectionalPropagationORT(
            backward_onnx_path=os.path.join(self.onnx_dir, 'backward_step.encmodel'),
            forward_onnx_path=os.path.join(self.onnx_dir, 'forward_step.encmodel'),
            backward_backbone_onnx_path=os.path.join(self.onnx_dir, 'backward_backbone.encmodel'),
            forward_backbone_onnx_path=os.path.join(self.onnx_dir, 'forward_backbone.encmodel'),
            fusion_onnx_path=os.path.join(self.onnx_dir, 'fusion.encmodel'),
            providers=providers, 
            provider_options=provider_options,
            sess_options=sess_options, 
            run_options=run_options
        )
        self.decoder_ort = DecoderORT(os.path.join(self.onnx_dir, 'decoder.encmodel'), providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)
        self._use_cupy = self.encoder_ort._use_cupy
        self.shrink_run_options = ppt_run_options(shrink_memory=True, use_cuda=self.encoder_ort._use_iobinding)

    def forward_bidirect_flow(self, masked_flows_f: np.ndarray, masked_flows_b: np.ndarray, masks: np.ndarray, shrink_memory: bool = False) -> tuple:
        if self._use_cupy:
            return self._forward_bidirect_flow_cupy(masked_flows_f, masked_flows_b, masks, shrink_memory)
        return self._forward_bidirect_flow_cpu(masked_flows_f, masked_flows_b, masks, shrink_memory)

    def _forward_bidirect_flow_cpu(self, masked_flows_f, masked_flows_b, masks, shrink_memory=False):
        b, t_flow, _, h, w = masked_flows_f.shape
        masks_f = masks[:, :-1, ...]
        masked_flows_f_in = masked_flows_f * (1 - masks_f)
        feat_mid_f, feat_e1_f, x_f = self.encoder_ort(masked_flows_f_in, masks_f)
        feat_mid_f_bt = feat_mid_f.transpose(0, 2, 1, 3, 4)
        feat_prop_f = self.prop_ort.forward(feat_mid_f_bt)
        feat_prop_f = feat_prop_f.reshape(-1, 128, h // 8, w // 8)
        _, c, _, h_f, w_f = feat_e1_f.shape
        feat_e1_f = feat_e1_f.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        _, c, _, h_f, w_f = x_f.shape
        x_f = x_f.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        flow_f, _ = self.decoder_ort(feat_prop_f, feat_e1_f, x_f)
        flow_f = flow_f.reshape(b, t_flow, 2, h, w)

        masks_b = masks[:, 1:, ...]
        masked_flows_b_in = masked_flows_b * (1 - masks_b)
        masked_flows_b_flipped = masked_flows_b_in[:, ::-1]
        masks_b_flipped = masks_b[:, ::-1]
        feat_mid_b, feat_e1_b, x_b = self.encoder_ort(masked_flows_b_flipped, masks_b_flipped)
        feat_mid_b_bt = feat_mid_b.transpose(0, 2, 1, 3, 4)
        feat_prop_b = self.prop_ort.forward(feat_mid_b_bt)
        feat_prop_b = feat_prop_b.reshape(-1, 128, h // 8, w // 8)
        _, c, _, h_f, w_f = feat_e1_b.shape
        feat_e1_b = feat_e1_b.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        _, c, _, h_f, w_f = x_b.shape
        x_b = x_b.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        flow_b_flipped, _ = self.decoder_ort(
            feat_prop_b,
            feat_e1_b,
            x_b,
            run_options=(self.shrink_run_options if shrink_memory else None),
        )
        flow_b = flow_b_flipped.reshape(b, t_flow, 2, h, w)[:, ::-1]
        return flow_f, flow_b

    def _forward_bidirect_flow_cupy(self, masked_flows_f, masked_flows_b, masks, shrink_memory=False):
        b, t_flow, _, h, w = masked_flows_f.shape
        mf_f = cp.asarray(masked_flows_f)
        mf_b = cp.asarray(masked_flows_b)
        masks_cp = cp.asarray(masks, dtype=cp.float32)

        # Forward flow
        masks_f = masks_cp[:, :-1, ...]
        masked_flows_f_in = mf_f * (1 - masks_f)
        feat_mid_f, feat_e1_f, x_f = self.encoder_ort(masked_flows_f_in, masks_f)
        feat_mid_f_bt = feat_mid_f.transpose(0, 2, 1, 3, 4)
        feat_prop_f = self.prop_ort.forward(feat_mid_f_bt)
        if isinstance(feat_prop_f, np.ndarray):
            feat_prop_f = cp.asarray(feat_prop_f)
        feat_prop_f = feat_prop_f.reshape(-1, 128, h // 8, w // 8)
        _, c, _, h_f, w_f = feat_e1_f.shape
        feat_e1_f = feat_e1_f.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        _, c, _, h_f, w_f = x_f.shape
        x_f = x_f.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        flow_f, _ = self.decoder_ort(feat_prop_f, feat_e1_f, x_f)
        if isinstance(flow_f, cp.ndarray):
            flow_f = cp.asnumpy(flow_f.reshape(b, t_flow, 2, h, w))
        else:
            flow_f = flow_f.reshape(b, t_flow, 2, h, w)

        # Backward flow
        masks_b = masks_cp[:, 1:, ...]
        masked_flows_b_in = mf_b * (1 - masks_b)
        masked_flows_b_flipped = masked_flows_b_in[:, ::-1]
        masks_b_flipped = masks_b[:, ::-1]
        feat_mid_b, feat_e1_b, x_b = self.encoder_ort(
            cp.ascontiguousarray(masked_flows_b_flipped),
            cp.ascontiguousarray(masks_b_flipped)
        )
        feat_mid_b_bt = feat_mid_b.transpose(0, 2, 1, 3, 4)
        feat_prop_b = self.prop_ort.forward(feat_mid_b_bt)
        if isinstance(feat_prop_b, np.ndarray):
            feat_prop_b = cp.asarray(feat_prop_b)
        feat_prop_b = feat_prop_b.reshape(-1, 128, h // 8, w // 8)
        _, c, _, h_f, w_f = feat_e1_b.shape
        feat_e1_b = feat_e1_b.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        _, c, _, h_f, w_f = x_b.shape
        x_b = x_b.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        flow_b_raw, _ = self.decoder_ort(
            feat_prop_b,
            feat_e1_b,
            x_b,
            run_options=(self.shrink_run_options if shrink_memory else None),
        )
        if isinstance(flow_b_raw, cp.ndarray):
            flow_b = cp.asnumpy(flow_b_raw.reshape(b, t_flow, 2, h, w)[:, ::-1])
        else:
            flow_b = flow_b_raw.reshape(b, t_flow, 2, h, w)[:, ::-1]

        return flow_f, flow_b

    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...]
        masks_backward = masks[:, 1:, ...]
        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1 - masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1 - masks_backward)
        return pred_flows_forward, pred_flows_backward
