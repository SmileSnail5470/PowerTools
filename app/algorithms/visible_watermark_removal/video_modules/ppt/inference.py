import os
import gc
import cv2
import time
import numpy as np
import scipy.ndimage
from PIL import Image
from app.algorithms.visible_watermark_removal.video_modules.ppt.propagation_transformer.run_propagation_transformer import PropagationTransformerORT
from app.algorithms.visible_watermark_removal.video_modules.ppt.raft.run_raft import RAFTBiONNX
from app.algorithms.visible_watermark_removal.video_modules.ppt.recurrent_flow_completion.run_flow_completion import RecurrentFlowCompleteORT
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


class PPTInferenceORT:
    def __init__(self, onnx_paths, resize_ratio=1.0, height=-1, width=-1, mask_dilation=4, ref_stride=10, neighbor_length=10, subvideo_length=80):
        self.onnx_paths = onnx_paths
        self.resize_ratio = resize_ratio
        self.height = height
        self.width = width
        self.mask_dilation = mask_dilation
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length

        self.raft_model = RAFTBiONNX(onnx_paths['raft'])
        self.fix_flow_complete = RecurrentFlowCompleteORT(onnx_dir=onnx_paths["recurrent_flow_complete"])
        self.ppt_pipeline = PropagationTransformerORT(onnx_paths["ppt"])
        self._use_cupy = getattr(self.ppt_pipeline, '_use_cupy', False) and _HAS_CUPY

    def release(self):
        if self.raft_model is not None:
            del self.raft_model
            self.raft_model = None
        if self.fix_flow_complete is not None:
            del self.fix_flow_complete
            self.fix_flow_complete = None
        if self.ppt_pipeline is not None:
            del self.ppt_pipeline
            self.ppt_pipeline = None
        gc.collect()

    def _imwrite(self, bgr_img, file_path, params=None):
        os.makedirs(os.path.abspath(os.path.dirname(file_path)), exist_ok=True)
        return cv2.imwrite(file_path, bgr_img, params)

    def _resize_frames(self, frames, size=None):
        out_size = frames[0].size
        if size is not None:
            process_size = (size[0] - size[0] % 8, size[1] - size[1] % 8)
        else:
            process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size or size is not None:
            frames = [f.resize(process_size) for f in frames]
        return frames, process_size, out_size

    def _read_frame(self, frame_root):
        frames = []
        for fr in sorted(os.listdir(frame_root)):
            frame = cv2.imread(os.path.join(frame_root, fr))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames, None, frames[0].size

    def _binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask

    def _read_mask(self, mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
        masks_img = [Image.open(mpath)] if os.path.isfile(mpath) else [Image.open(os.path.join(mpath, mp)) for mp in sorted(os.listdir(mpath))]
        masks_dilated, flow_masks = [], []
        for mask_img in masks_img:
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            mask_img = np.array(mask_img.convert('L'))
            f_mask = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8) if flow_mask_dilates > 0 else self._binary_mask(mask_img).astype(np.uint8)
            flow_masks.append(Image.fromarray(f_mask * 255))
            m_mask = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8) if mask_dilates > 0 else self._binary_mask(mask_img).astype(np.uint8)
            masks_dilated.append(Image.fromarray(m_mask * 255))
        if len(masks_img) == 1:
            flow_masks *= length
            masks_dilated *= length
        return flow_masks, masks_dilated

    def _get_ref_index(self, mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def inference(self, input_frames_dir, masks_dir, output_dir, debug=False):
        frames, _, size = self._read_frame(frame_root=input_frames_dir)
        if self.width != -1 and self.height != -1:
            size = (self.width, self.height)
        if self.resize_ratio != 1.0:
            size = (int(self.resize_ratio * size[0]), int(self.resize_ratio * size[1]))

        frames, size, out_size = self._resize_frames(frames, size)
        os.makedirs(output_dir, exist_ok=True)

        frames_len = len(frames)
        flow_masks, masks_dilated = self._read_mask(masks_dir, frames_len, size, flow_mask_dilates=self.mask_dilation, mask_dilates=self.mask_dilation)

        ori_frames = [np.array(f).astype(np.uint8) for f in frames]
        frames_np = np.ascontiguousarray(np.expand_dims(np.stack(ori_frames, axis=0).transpose(0, 3, 1, 2), axis=0).astype(np.float32) / 255.0 * 2.0 - 1.0)
        flow_masks_np = np.ascontiguousarray(np.expand_dims(np.stack([np.array(m) for m in flow_masks], axis=0), axis=(0, 2)).astype(np.float32) / 255.0)
        masks_dilated_np = np.ascontiguousarray(np.expand_dims(np.stack([np.array(m) for m in masks_dilated], axis=0), axis=(0, 2)).astype(np.float32) / 255.0)

        video_length = frames_np.shape[1]
        if debug:
            print(f'\nProcessing pure ONNX Pipeline: [{video_length} frames]...')
            start_time = time.time()

        raft_scale = 0.5
        short_clip_len = 12 if max(frames_np.shape) <= 1280 else 6
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                sub_frames = frames_np[:, f:end_f] if f == 0 else frames_np[:, f - 1:end_f]
                flows_f, flows_b = self.raft_model.forward(sub_frames, scale_factor=raft_scale)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
            gt_flows_bi = (np.concatenate(gt_flows_f_list, axis=1), np.concatenate(gt_flows_b_list, axis=1))
        else:
            gt_flows_bi = self.raft_model.forward(frames_np, scale_factor=raft_scale)
        del self.raft_model
        self.raft_model = None
        gc.collect()
        if debug:
            print(f'RAFT ONNX Inference Cost: {time.time() - start_time:.4f}s')
            start_time = time.time()

        flow_length = gt_flows_bi[0].shape[1]
        if flow_length > self.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, self.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.subvideo_length)
                sub_f, sub_b = self.fix_flow_complete.forward_bidirect_flow(
                    gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f], flow_masks_np[:, s_f:e_f + 1]
                )
                sub_comb_f, sub_comb_b = self.fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), (sub_f, sub_b), flow_masks_np[:, s_f:e_f + 1]
                )
                end_idx = e_f - s_f - pad_len_e
                pred_flows_f.append(sub_comb_f[:, pad_len_s:end_idx])
                pred_flows_b.append(sub_comb_b[:, pad_len_s:end_idx])
            pred_flows_bi = (np.concatenate(pred_flows_f, axis=1), np.concatenate(pred_flows_b, axis=1))
        else:
            pred_flows_f, pred_flows_b = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi[0], gt_flows_bi[1], flow_masks_np)
            pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, (pred_flows_f, pred_flows_b), flow_masks_np)
        del self.fix_flow_complete, gt_flows_bi
        self.fix_flow_complete = None
        gc.collect()
        if debug:
            print(f'Flow Completion ONNX Inference Cost: {time.time() - start_time:.4f}s')
            start_time = time.time()

        masked_frames_np = frames_np * (1.0 - masks_dilated_np)
        subvideo_length_img_prop = min(100, self.subvideo_length)
        if video_length > subvideo_length_img_prop:
            updated_frames_list, updated_masks_list = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                sub_flows_bi = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = self.ppt_pipeline.img_propagation(
                    masked_frames_np[:, s_f:e_f], sub_flows_bi, masks_dilated_np[:, s_f:e_f]
                )
                updated_frames_sub = frames_np[:, s_f:e_f] * (1.0 - masks_dilated_np[:, s_f:e_f]) + prop_imgs_sub * masks_dilated_np[:, s_f:e_f]
                end_idx = e_f - s_f - pad_len_e
                updated_frames_list.append(updated_frames_sub[:, pad_len_s:end_idx])
                updated_masks_list.append(updated_local_masks_sub[:, pad_len_s:end_idx])
            updated_frames = np.concatenate(updated_frames_list, axis=1)
            updated_masks = np.concatenate(updated_masks_list, axis=1)
        else:
            prop_imgs, updated_local_masks = self.ppt_pipeline.img_propagation(masked_frames_np, pred_flows_bi, masks_dilated_np)
            updated_frames = frames_np * (1.0 - masks_dilated_np) + prop_imgs * masks_dilated_np
            updated_masks = updated_local_masks
        if debug:
            print(f'Image Propagation ONNX Inference Cost: {time.time() - start_time:.4f}s')
            start_time = time.time()

        del masked_frames_np, frames_np, flow_masks_np
        gc.collect()

        if self._use_cupy:
            self._transformer_loop_cupy(video_length, updated_frames, updated_masks, masks_dilated_np, pred_flows_bi, ori_frames, out_size, output_dir)
        else:
            self._transformer_loop_cpu(video_length, updated_frames, updated_masks, masks_dilated_np, pred_flows_bi, ori_frames, out_size, output_dir)

        if debug:
            print(f'Propagation Transformer ONNX Inference Cost: {time.time() - start_time:.4f}s')
        del self.ppt_pipeline
        self.ppt_pipeline = None
        gc.collect()

    def _transformer_loop_cpu(self, video_length, updated_frames, updated_masks, masks_dilated_np, pred_flows_bi, ori_frames, out_size, output_dir):
        comp_frames = [None] * video_length
        neighbor_stride = self.neighbor_length // 2 if max(updated_frames.shape) > 540 else self.neighbor_length
        ref_num = self.subvideo_length // self.ref_stride if video_length > self.subvideo_length else -1
        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
            ref_ids = self._get_ref_index(f, neighbor_ids, video_length, self.ref_stride, ref_num)
            selected_imgs = np.ascontiguousarray(updated_frames[:, neighbor_ids + ref_ids])
            selected_masks = np.ascontiguousarray(masks_dilated_np[:, neighbor_ids + ref_ids])
            selected_update_masks = np.ascontiguousarray(updated_masks[:, neighbor_ids + ref_ids])
            selected_pred_flows_bi = (
                np.ascontiguousarray(pred_flows_bi[0][:, neighbor_ids[:-1]]),
                np.ascontiguousarray(pred_flows_bi[1][:, neighbor_ids[:-1]])
            )
            l_t = len(neighbor_ids)
            pred_img = self.ppt_pipeline.forward(
                selected_imgs, selected_pred_flows_bi[0], selected_pred_flows_bi[1],
                selected_masks, selected_update_masks, num_local_frames=l_t
            )
            pred_img = np.clip((pred_img + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            binary_masks = masks_dilated_np[0, neighbor_ids].transpose(0, 2, 3, 1)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img_pred_hwc = pred_img[0, i].transpose(1, 2, 0)
                img = (img_pred_hwc * binary_masks[i] + ori_frames[idx] * (1.0 - binary_masks[i])).astype(np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5).astype(np.uint8)
        for idx in range(video_length):
            f = cv2.resize(comp_frames[idx], out_size, interpolation=cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            self._imwrite(f, os.path.join(output_dir, str(idx).zfill(6) + '.png'))

    def _transformer_loop_cupy(self, video_length, updated_frames, updated_masks, masks_dilated_np, pred_flows_bi, ori_frames, out_size, output_dir):
        uf_gpu = cp.asarray(updated_frames)
        um_gpu = cp.asarray(updated_masks)
        md_gpu = cp.asarray(masks_dilated_np)
        pf_f_gpu = cp.asarray(pred_flows_bi[0])
        pf_b_gpu = cp.asarray(pred_flows_bi[1])
        ori_gpu = cp.asarray(np.stack(ori_frames, axis=0).astype(np.float32))  # (T, H, W, C)

        comp_frames_gpu = [None] * video_length
        neighbor_stride = self.neighbor_length // 2 if max(updated_frames.shape) > 540 else self.neighbor_length
        ref_num = self.subvideo_length // self.ref_stride if video_length > self.subvideo_length else -1
        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
            ref_ids = self._get_ref_index(f, neighbor_ids, video_length, self.ref_stride, ref_num)
            all_ids = neighbor_ids + ref_ids
            n_ids_arr = neighbor_ids[:-1]
            selected_imgs = cp.ascontiguousarray(uf_gpu[:, all_ids])
            selected_masks = cp.ascontiguousarray(md_gpu[:, all_ids])
            selected_update_masks = cp.ascontiguousarray(um_gpu[:, all_ids])
            selected_flows_f = cp.asnumpy(cp.ascontiguousarray(pf_f_gpu[:, n_ids_arr]))
            selected_flows_b = cp.asnumpy(cp.ascontiguousarray(pf_b_gpu[:, n_ids_arr]))

            l_t = len(neighbor_ids)
            pred_img = self.ppt_pipeline.forward(
                cp.asnumpy(selected_imgs), selected_flows_f, selected_flows_b,
                cp.asnumpy(selected_masks), cp.asnumpy(selected_update_masks), num_local_frames=l_t
            )
            pred_img_gpu = cp.asarray(pred_img)
            pred_img_gpu = cp.clip((pred_img_gpu + 1.0) / 2.0 * 255.0, 0, 255).astype(cp.uint8)

            binary_masks = md_gpu[0, neighbor_ids].transpose(0, 2, 3, 1)  # (len, H, W, 1)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img_pred_hwc = pred_img_gpu[0, i].transpose(1, 2, 0).astype(cp.float32)
                mask_i = binary_masks[i]
                img = (img_pred_hwc * mask_i + ori_gpu[idx] * (1.0 - mask_i)).astype(cp.uint8)
                if comp_frames_gpu[idx] is None:
                    comp_frames_gpu[idx] = img
                else:
                    comp_frames_gpu[idx] = (comp_frames_gpu[idx].astype(cp.float32) * 0.5 + img.astype(cp.float32) * 0.5).astype(cp.uint8)
        for idx in range(video_length):
            f = cp.asnumpy(comp_frames_gpu[idx])
            f = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            self._imwrite(f, os.path.join(output_dir, str(idx).zfill(6) + '.png'))
