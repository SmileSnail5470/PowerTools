import numpy as np
import onnxruntime as ort
from app.algorithms import general_inference_session

class SparseAttentionCoreORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options

    def __call__(self, x):
        outputs = self.session.run(None, {self.input_names[0]: x}, run_options=self.run_options)
        return outputs[0], outputs[1], outputs[2]  # win_q, win_k, win_v
    
    def __del__(self):
        self.session = None


class AttentionComputationORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options

    def __call__(self, q, k, v):
        feed_dict = {
            self.input_names[0]: q,
            self.input_names[1]: k,
            self.input_names[2]: v,
        }
        outputs = self.session.run(None, feed_dict, run_options=self.run_options)
        return outputs[0]
    
    def __del__(self):
        self.session = None


class MLPComputationORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options

    def __call__(self, x, enc_feat):
        feed_dict = {
            self.input_names[0]: x,
            self.input_names[1]: enc_feat,
        }
        outputs = self.session.run(None, feed_dict, run_options=self.run_options)
        return outputs[0]
    
    def __del__(self):
        self.session = None
    

class OutputProjectionORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options

    def __call__(self, att_out, x):
        feed_dict = {
            self.input_names[0]: att_out,
            self.input_names[1]: x,
        }
        outputs = self.session.run(None, feed_dict, run_options=self.run_options)
        return outputs[0]
    
    def __del__(self):
        self.session = None


class NormComputationORT:
    def __init__(self, onnx_path, providers, provider_options=None, sess_options=None, run_options=None):
        self.session = general_inference_session(onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.run_options = run_options

    def __call__(self, x):
        feed_dict = {self.input_names[0]: x}
        outputs = self.session.run(None, feed_dict, run_options=self.run_options)
        return outputs[0]
    
    def __del__(self):
        self.session = None


class TemporalSparseTransformerBlockORT:
    def __init__(self, depths, window_size, onnx_paths, providers, provider_options=None, sess_options=None, run_options=None):
        self.depths = depths
        self.window_size = window_size
        
        self.core_layers = [SparseAttentionCoreORT(onnx_paths["core"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.attn_layers = [AttentionComputationORT(onnx_paths["attn"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.proj_layers = [OutputProjectionORT(onnx_paths["proj"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.norm1_layers = [NormComputationORT(onnx_paths["norm1"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.norm2_layers = [NormComputationORT(onnx_paths["norm2"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]
        self.mlp_layers = [MLPComputationORT(onnx_paths["mlp"][i], providers, provider_options, sess_options, run_options) for i in range(self.depths)]

    def __del__(self):
        self.core_layers = None
        self.attn_layers = None
        self.proj_layers = None
        self.norm1_layers = None
        self.norm2_layers = None
        self.mlp_layers = None

    def _numpy_max_pool2d_and_sum(self, mask, new_h, new_w, n_wh, n_ww):
        B, T, H, W, _ = mask.shape
        pad_h = new_h - H
        pad_w = new_w - W
        if pad_h > 0 or pad_w > 0:
            mask_pad = np.pad(mask, ((0,0), (0,0), (0, pad_h), (0, pad_w), (0,0)), mode='constant', constant_values=0)
        else:
            mask_pad = mask
            
        mask_pad = np.squeeze(mask_pad, axis=-1)  # [B, T, new_h, new_w]
        w_h, w_w = self.window_size
        blocked = mask_pad.reshape(B, T, n_wh, w_h, n_ww, w_w)
        pooled = blocked.max(axis=(3, 5))
        
        pooled_flat = pooled.reshape(B, T, n_wh * n_ww)
        mask_sum = pooled_flat.sum(axis=1)  # [B, n_wh * n_ww]
        return mask_sum

    def __call__(self, x, enc_feat, l_mask, t_dilation=2):
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        B, T, H, W, C = x.shape
        w_h, w_w = self.window_size
        w_h_w_w = w_h * w_w
        n_head = self.core_layers[0].session.get_outputs()[0].shape[2]
        c_head = C // n_head

        n_wh = int(np.ceil(H / w_h)) if 'h' in locals() else int(np.ceil(H / w_h))
        n_ww = int(np.ceil(W / w_w)) if 'w' in locals() else int(np.ceil(W / w_w))
        new_h = n_wh * w_h
        new_w = n_ww * w_w

        t_indices = []
        for _ in range(self.depths // t_dilation):
            for i in range(t_dilation):
                t_indices.append(np.arange(i, T, t_dilation))
        mask_sum = self._numpy_max_pool2d_and_sum(l_mask, new_h, new_w, n_wh, n_ww)

        for i in range(self.depths):
            shortcut = x
            norm_x = self.norm1_layers[i](x)
            win_q, win_k, win_v = self.core_layers[i](norm_x)

            out_layer = np.zeros_like(win_q)
            curr_t_ind = t_indices[i]
            for b_idx in range(B):
                layer_mask_b = mask_sum[b_idx]
                mask_ind_i = np.nonzero(layer_mask_b != 0)[0]
                mask_n = len(mask_ind_i)
                if mask_n > 0:
                    win_q_t = win_q[b_idx, mask_ind_i]
                    win_k_t = win_k[b_idx, mask_ind_i]
                    win_v_t = win_v[b_idx, mask_ind_i]                  
                    win_k_t = win_k_t[:, :, curr_t_ind, :, :]
                    win_v_t = win_v_t[:, :, curr_t_ind, :, :]                   
                    q_flat = win_q_t.reshape(mask_n, n_head, T * w_h_w_w, c_head)
                    k_flat = win_k_t.reshape(mask_n, n_head, len(curr_t_ind) * win_k_t.shape[3], c_head)
                    v_flat = win_v_t.reshape(mask_n, n_head, len(curr_t_ind) * win_v_t.shape[3], c_head)                   
                    y_t = self.attn_layers[i](q_flat, k_flat, v_flat)
                    out_layer[b_idx, mask_ind_i] = y_t.reshape(mask_n, n_head, T, w_h_w_w, c_head)

                unmask_ind_i = np.nonzero(layer_mask_b == 0)[0]
                unmask_n = len(unmask_ind_i)
                if unmask_n > 0:
                    win_q_s = win_q[b_idx, unmask_ind_i]
                    win_k_s = win_k[b_idx, unmask_ind_i, :, :, :w_h_w_w, :]
                    win_v_s = win_v[b_idx, unmask_ind_i, :, :, :w_h_w_w, :]
                    q_flat_s = win_q_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    k_flat_s = win_k_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    v_flat_s = win_v_s.reshape(unmask_n, n_head, T * w_h_w_w, c_head)
                    
                    y_s = self.attn_layers[i](q_flat_s, k_flat_s, v_flat_s)
                    out_layer[b_idx, unmask_ind_i] = y_s.reshape(unmask_n, n_head, T, w_h_w_w, c_head)

            att_x = self.proj_layers[i](out_layer, x)
            x = shortcut + att_x
            shortcut_mlp = x
            norm_y = self.norm2_layers[i](x)
            mlp_out = self.mlp_layers[i](norm_y.reshape(B, T * H * W, C), enc_feat)
            x = shortcut_mlp + mlp_out.reshape(B, T, H, W, C)
            del win_q, win_k, win_v, norm_x, norm_y, att_x, mlp_out
        return x