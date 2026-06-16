import os
import platform
import sys
import numpy as np
import onnxruntime as ort
from app.algorithms import ORTEnvironment
ORTEnvironment.initialize()
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.encoder import EncoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.decoder import DecoderORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.bidirectional_propagation import BidirectionalPropagationORT


class RecurrentFlowCompleteORT:
    def __init__(self, onnx_dir):
        self.onnx_dir = onnx_dir
        self.encoder_ort = None
        self.prop_ort = None
        self.decoder_ort = None
        self.prepare_models()

    def _get_session_options(self):
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.use_env_allocators", "1")
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def _get_providers(self):
        available = ort.get_available_providers()
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        elif "CUDAExecutionProvider" in available and self._hash_cuda_gpu():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        return providers, provider_options
    
    def _hash_cuda_gpu(self):
        if platform.system() != "Windows":
            return True
        cuda_path = r"C:\Program Files\NVIDIA Corporation"
        if os.path.exists(cuda_path):
            return True
        return False
    
    def __del__(self):
        if hasattr(self, 'encoder_ort'):
            del self.encoder_ort
        if hasattr(self, 'prop_ort'):
            del self.prop_ort
        if hasattr(self, 'decoder_ort'):
            del self.decoder_ort
    
    def prepare_models(self):
        if self.encoder_ort is not None and self.prop_ort is not None and self.decoder_ort is not None:
            return
        sess_options = self._get_session_options()
        providers, provider_options = self._get_providers()
        run_options = ort.RunOptions()
        self.encoder_ort = EncoderORT(os.path.join(self.onnx_dir, 'encoder.encmodel'), providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)
        self.prop_ort = BidirectionalPropagationORT(
            backward_onnx_path = os.path.join(self.onnx_dir, 'backward_step.encmodel'),
            forward_onnx_path = os.path.join(self.onnx_dir, 'forward_step.encmodel'),
            backward_backbone_onnx_path = os.path.join(self.onnx_dir, 'backward_backbone.encmodel'),
            forward_backbone_onnx_path = os.path.join(self.onnx_dir, 'forward_backbone.encmodel'),
            fusion_onnx_path = os.path.join(self.onnx_dir, 'fusion.encmodel'),
            providers=providers,
            provider_options=provider_options,
            sess_options=sess_options,
            run_options=run_options
        )
        self.decoder_ort = DecoderORT(os.path.join(self.onnx_dir, 'decoder.encmodel'), providers=providers, provider_options=provider_options, sess_options=sess_options, run_options=run_options)

    def forward_bidirect_flow(self, masked_flows_f: np.ndarray, masked_flows_b: np.ndarray, masks: np.ndarray) -> tuple:
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
        flow_f, edge_f = self.decoder_ort(feat_prop_f, feat_e1_f, x_f)
        flow_f = flow_f.reshape(b, t_flow, 2, h, w)

        masks_b = masks[:, 1:, ...]
        masked_flows_b_masked = masked_flows_b * (1 - masks_b)
        masked_flows_b_flipped = masked_flows_b_masked[:, ::-1]
        masks_b_flipped = masks_b[:, ::-1]
        feat_mid_b, feat_e1_b, x_b = self.encoder_ort(masked_flows_b_flipped, masks_b_flipped)
        feat_mid_b_bt = feat_mid_b.transpose(0, 2, 1, 3, 4)
        feat_prop_b = self.prop_ort.forward(feat_mid_b_bt)
        feat_prop_b = feat_prop_b.reshape(-1, 128, h // 8, w // 8)
        _, c, _, h_f, w_f = feat_e1_b.shape
        feat_e1_b = feat_e1_b.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        _, c, _, h_f, w_f = x_b.shape
        x_b = x_b.transpose(0, 2, 1, 3, 4).reshape(-1, c, h_f, w_f)
        flow_b_flipped, edge_b = self.decoder_ort(feat_prop_b, feat_e1_b, x_b)
        flow_b = flow_b_flipped.reshape(b, t_flow, 2, h, w)[:, ::-1]
        return flow_f, flow_b

    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...]
        masks_backward = masks[:, 1:, ...]
        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1-masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1-masks_backward)
        return pred_flows_forward, pred_flows_backward