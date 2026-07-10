import logging
import cv2
import numpy as np
import app.utils.ffmpeg as ffmpeg
import shutil
import tempfile
import os
from PIL import Image
from pathlib import Path
from app.algorithms.visible_watermark_removal.watermark_removal_image import ImageWatermarkRemove, WatermarkSegment
from app.algorithms.visible_watermark_removal.video_modules.ppt.inference import PPTInferenceORT
from app.algorithms.tracker.mask_tracker_nano import MaskTrackerNano


class VideoWatermarkRemover:
    def __init__(self):
        self.video_models_name = ["ppt"]

    def _save_mask_visualization(self, img_path, mask: np.ndarray, file_path: str):
        img = cv2.imread(img_path)
        overlay = img.copy()
        color = (255, 0, 0)
        b, g, r_ = color
        mask = mask > 0
        overlay[mask] = (overlay[mask] * 0.55 + np.array([b, g, r_]) * 0.45).astype(np.uint8)
        cv2.imwrite(file_path, overlay)

    def _expand_bbox_keep_center(self, xmin: int, ymin: int, xmax: int, ymax: int, img_w: int, img_h: int, pad: int = 120):
        xmin_new = xmin - pad
        xmax_new = xmax + pad
        if xmin_new < 0:
            extra = -xmin_new
            xmin_new = 0
            xmax_new = min(img_w, xmax_new + extra)
        if xmax_new > img_w:
            extra = xmax_new - img_w
            xmax_new = img_w
            xmin_new = max(0, xmin_new - extra)

        ymin_new = ymin - pad
        ymax_new = ymax + pad
        if ymin_new < 0:
            extra = -ymin_new
            ymin_new = 0
            ymax_new = min(img_h, ymax_new + extra)
        if ymax_new > img_h:
            extra = ymax_new - img_h
            ymax_new = img_h
            ymin_new = max(0, ymin_new - extra)
        return (int(xmin_new), int(ymin_new), int(xmax_new), int(ymax_new))

    def _get_roi(self, masks_dir: str) -> tuple|None:
        mask_paths = os.listdir(masks_dir)
        if not mask_paths:
            return None
        global_xmin, global_ymin = float('inf'), float('inf')
        global_xmax, global_ymax = float('-inf'), float('-inf')
        for mask_path in mask_paths:
            mask = cv2.imread(os.path.join(masks_dir, mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask image: {os.path.join(masks_dir, mask_path)}")
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x = coords[:, 0, 0]
                y = coords[:, 0, 1]
                global_xmin = min(global_xmin, x.min())
                global_ymin = min(global_ymin, y.min())
                global_xmax = max(global_xmax, x.max())
                global_ymax = max(global_ymax, y.max())
        if global_xmin == float('inf'):
            return None
        return self._expand_bbox_keep_center(int(global_xmin), int(global_ymin), int(global_xmax)+1, int(global_ymax)+1, mask.shape[1], mask.shape[0])

    def _crop_frames_and_masks(self, input_frames_dir: str, masks_dir: str, bbox: tuple) -> tuple:
        new_input_dir = os.path.join(os.path.dirname(input_frames_dir), "{0}_cropped".format(os.path.basename(input_frames_dir)))
        new_masks_dir = os.path.join(os.path.dirname(masks_dir), "{0}_cropped".format(os.path.basename(masks_dir)))
        os.makedirs(new_input_dir, exist_ok=True)
        os.makedirs(new_masks_dir, exist_ok=True)
        if bbox is None:
            return input_frames_dir, masks_dir
        xmin, ymin, xmax, ymax = bbox
        frame_paths = [os.path.join(input_frames_dir, frame_file) for frame_file in os.listdir(input_frames_dir)]
        for img_path in frame_paths:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read frame image: {img_path}")
            h, w = img.shape[:2]
            x1, y1 = max(0, xmin), max(0, ymin)
            x2, y2 = min(w, xmax), min(h, ymax)
            cropped_img = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(new_input_dir, os.path.basename(img_path)), cropped_img)
        mask_paths = [os.path.join(masks_dir, mask_file) for mask_file in os.listdir(masks_dir)]
        for m_path in mask_paths:
            mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask image: {m_path}")
            h, w = mask.shape[:2]
            x1, y1 = max(0, xmin), max(0, ymin)
            x2, y2 = min(w, xmax), min(h, ymax)
            cropped_mask = mask[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(new_masks_dir, os.path.basename(m_path)), cropped_mask)
        return new_input_dir, new_masks_dir
    
    def _past_frames_back(self, input_frames_dir: str, cropped_out_dir: str, out_path_dir: str, bbox: tuple):
        if bbox is None:
            for p in os.listdir(cropped_out_dir):
                shutil.copy2(os.path.join(cropped_out_dir, p), os.path.join(out_path_dir, os.path.basename(p)))
            return
        xmin, ymin, xmax, ymax = bbox
        cropped_paths = sorted([os.path.join(cropped_out_dir, item) for item in os.listdir(cropped_out_dir)])
        for c_path in cropped_paths:
            cropped_img = cv2.imread(c_path)
            if cropped_img is None:
                 raise ValueError(f"Failed to read frame image: {c_path}")
            original = cv2.imread(os.path.join(input_frames_dir, os.path.basename(c_path)))
            if original is None:
                raise ValueError(f"Failed to read frame image: {os.path.join(input_frames_dir, os.path.basename(c_path))}")
            canvas = original
            canvas[ymin:ymax, xmin:xmax] = cropped_img
            cv2.imwrite(os.path.join(out_path_dir, os.path.basename(c_path)), canvas)

    def _postprocess_mask(self, mask):
        h, w = mask.shape[-2]
        mask_image = np.zeros((h, w), dtype=np.uint8)
        for bbox in self.watermark_boxes:
            xmin, ymin, xmax, ymax = bbox
            mask_image[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
        new_mask = mask_image
        return new_mask

    def _prepare_masks(
            self, 
            tmp_mask_dir: str|Path, 
            frame_mask_map: dict, 
            mask_path: str, 
            use_cache_mask: bool, 
            watermark_type: str,
            ai_detect_type,
            ai_interactive_type,
            ai_interactive_prompt,
            ai_interactive_boxes,
            watermark_confidence: float,
            frame_files: list,
            sr_segment_onnx_path,
            pt_segment_onnx_path,
            text_detection_onnx_path,
            yolo_detection_onnx_path,
            segment_onnx_dir,
            tacker_onnx_dir,
            **kwargs
        ):
        need_postprocess_mask = self.watermark_boxes and not (ai_detect_type == "ai_interactive_detect" and ai_interactive_type == "space_detect")
        if use_cache_mask and not mask_path:
            # 静态水印，AI检测
            mask = WatermarkSegment(watermark_type, ai_detect_type, ai_interactive_type, ai_interactive_prompt, ai_interactive_boxes, watermark_confidence).segment(
                image_path=str(frame_files[0]),
                sr_onnx_path=sr_segment_onnx_path,
                pt_onnx_path=pt_segment_onnx_path,
                text_detection_onnx_path=text_detection_onnx_path,
                yolo_detection_onnx_path=yolo_detection_onnx_path,
                segment_onnx_dir=segment_onnx_dir,
                **kwargs
            )
            if need_postprocess_mask:
                mask = self._postprocess_mask(mask)
            tmp_mask_path = os.path.join(str(tmp_mask_dir), "mask.png")
            Image.fromarray(mask).convert("L").save(tmp_mask_path)
            for frame_file in frame_files:
                frame_mask_map[str(frame_file)] = tmp_mask_path
        elif mask_path and not os.path.isdir(mask_path):
            # 静态水印，人工提供水印
            tmp_mask_path = os.path.join(str(tmp_mask_dir), "mask.png")
            shutil.copy2(mask_path, tmp_mask_path)
            for frame_file in frame_files:
                frame_mask_map[str(frame_file)] = tmp_mask_path
        elif mask_path and os.path.isdir(mask_path):
            # 人工提供所有帧水印
            masks = os.listdir(mask_path)
            masks.sort()
            for frame_file, tmp_mask_path in zip(frame_files, masks):
                frame_mask_map[str(frame_file)] = os.path.join(mask_path, tmp_mask_path)
            tmp_mask_dir = mask_path
        else:
            # 动态水印，ai 检测
            if self.video_tracking_data is None:
                for i, frame_file in enumerate(frame_files):
                    mask = WatermarkSegment(watermark_type, ai_detect_type, ai_interactive_type, ai_interactive_prompt, ai_interactive_boxes, watermark_confidence).segment(
                        image_path=str(frame_file),
                        sr_onnx_path=sr_segment_onnx_path,
                        pt_onnx_path=pt_segment_onnx_path,
                        text_detection_onnx_path=text_detection_onnx_path,
                        yolo_detection_onnx_path=yolo_detection_onnx_path,
                        segment_onnx_dir=segment_onnx_dir,
                        **kwargs
                    )
                    if need_postprocess_mask:
                        mask = self._postprocess_mask(mask)
                    tmp_mask_path = os.path.join(str(tmp_mask_dir), frame_file.name)
                    Image.fromarray(mask).convert("L").save(tmp_mask_path)
                    frame_mask_map[str(frame_file)] = tmp_mask_path
            else:
                keyframes = self.video_tracking_data.get("keyframes")
                end_frame = self.video_tracking_data.get("end_frame")
                logging.getLogger("subprocess").info(f"Start track mask with keyframes {keyframes} and end frame {end_frame}")
                total_frames = len(frame_files)
                effective_end = end_frame if end_frame else total_frames
                for seg_idx, kf in enumerate(keyframes):
                    if seg_idx + 1 < len(keyframes):
                        seg_end = keyframes[seg_idx + 1]
                    else:
                        seg_end = effective_end + 1
                    kf_idx = kf - 1
                    if kf_idx < 0 or kf_idx >= total_frames:
                        continue
                    keyframe_file = frame_files[kf_idx]
                    mask = WatermarkSegment(watermark_type, ai_detect_type, ai_interactive_type, ai_interactive_prompt, ai_interactive_boxes, watermark_confidence).segment(
                        image_path=str(keyframe_file),
                        sr_onnx_path=sr_segment_onnx_path,
                        pt_onnx_path=pt_segment_onnx_path,
                        text_detection_onnx_path=text_detection_onnx_path,
                        yolo_detection_onnx_path=yolo_detection_onnx_path,
                        segment_onnx_dir=segment_onnx_dir,
                        **kwargs
                    )
                    if need_postprocess_mask:
                        mask = self._postprocess_mask(mask)
                    tmp_mask_path = os.path.join(str(tmp_mask_dir), keyframe_file.name)
                    Image.fromarray(mask).convert("L").save(tmp_mask_path)
                    frame_mask_map[str(keyframe_file)] = tmp_mask_path
                    seg_frame_start = kf_idx + 1
                    seg_frame_end = min(seg_end - 1, total_frames)
                    if seg_frame_start >= seg_frame_end:
                        continue
                    current_track_start = seg_frame_start
                    current_ref_mask_path = tmp_mask_path
                    current_ref_frame = keyframe_file
                    while current_track_start < seg_frame_end:
                        remaining_frames = frame_files[current_track_start:seg_frame_end]
                        if not remaining_frames:
                            break
                        seg_frames_dir = tempfile.mkdtemp(prefix="track_seg_")
                        try:
                            all_seg_files = [current_ref_frame] + remaining_frames
                            for frame_file in all_seg_files:
                                dst = os.path.join(seg_frames_dir, frame_file.name)
                                shutil.copy2(str(frame_file), dst)
                            tracker_instance = MaskTrackerNano(tacker_onnx_dir=tacker_onnx_dir, score_threshold=0.5)
                            fail_frame_idx = tracker_instance.inference(current_ref_mask_path, seg_frames_dir)
                            if fail_frame_idx == -1:
                                for frame_file in remaining_frames:
                                    tracked_mask_path = os.path.join(str(tmp_mask_dir), frame_file.name)
                                    if os.path.exists(tracked_mask_path):
                                        frame_mask_map[str(frame_file)] = tracked_mask_path
                                break
                            else:
                                successfully_tracked_count = fail_frame_idx - 1
                                for i in range(successfully_tracked_count):
                                    if i < len(remaining_frames):
                                        frame_file = remaining_frames[i]
                                        tracked_mask_path = os.path.join(str(tmp_mask_dir), frame_file.name)
                                        if os.path.exists(tracked_mask_path):
                                            frame_mask_map[str(frame_file)] = tracked_mask_path
                                failed_frame_offset = successfully_tracked_count
                                if failed_frame_offset >= len(remaining_frames):
                                    break
                                failed_frame_file = remaining_frames[failed_frame_offset]
                                logging.getLogger("subprocess").warning(f"Tracking failed at frame {failed_frame_file.name}, re-detecting as new keyframe")
                                new_keyframe_mask = WatermarkSegment(watermark_type, ai_detect_type, ai_interactive_type, ai_interactive_prompt, ai_interactive_boxes, watermark_confidence).segment(
                                    image_path=str(failed_frame_file),
                                    sr_onnx_path=sr_segment_onnx_path,
                                    pt_onnx_path=pt_segment_onnx_path,
                                    text_detection_onnx_path=text_detection_onnx_path,
                                    yolo_detection_onnx_path=yolo_detection_onnx_path,
                                    segment_onnx_dir=segment_onnx_dir,
                                    **kwargs
                                )
                                if need_postprocess_mask:
                                    new_keyframe_mask = self._postprocess_mask(new_keyframe_mask)
                                new_keyframe_mask_path = os.path.join(str(tmp_mask_dir), failed_frame_file.name)
                                Image.fromarray(new_keyframe_mask).convert("L").save(new_keyframe_mask_path)
                                frame_mask_map[str(failed_frame_file)] = new_keyframe_mask_path
                                current_ref_mask_path = new_keyframe_mask_path
                                current_ref_frame = failed_frame_file
                                current_track_start = current_track_start + failed_frame_offset + 1
                        finally:
                            shutil.rmtree(seg_frames_dir, ignore_errors=True)

    def _merge_processed_frames(self, processed_frames_dir, has_audio, fps, input_video_path, output_video_path):
        video_input = ffmpeg.input(
            str(processed_frames_dir / '%06d.png'),
            pixel_format='rgb24',
            framerate=fps
        )
    
        output_kwargs = {
            'vcodec': 'libx264',
            'pix_fmt': 'yuv420p',
            'start_number': 0
        }
        
        # 如果原视频有音频，保留音频
        audio_input = None
        if has_audio:
            audio_input = ffmpeg.input(input_video_path).audio
        
        # 组合视频和音频（如果有）
        if audio_input:
            stream = ffmpeg.output(
                video_input,
                audio_input,
                output_video_path,
                **output_kwargs
            ).global_args("-hide_banner", "-loglevel", "error")
        else:
            stream = ffmpeg.output(
                video_input,
                output_video_path,
                **output_kwargs
            ).global_args("-hide_banner", "-loglevel", "error")
        ffmpeg.run(stream, overwrite_output=True, quiet=False)

    def _image_model_inpainting(self, args: dict):
        processed_frames_dir = args.get("processed_frames_dir")
        frame_mask_map = args.get("frame_mask_map")
        sr_segment_onnx_path = args.get("sr_segment_onnx_path")
        pt_segment_onnx_path = args.get("pt_segment_onnx_path")
        pt_inpaint_onnx_path = args.get("pt_inpaint_onnx_path")
        cf_onnx_path = args.get("cf_onnx_path")
        lama_onnx_path = args.get("lama_onnx_path")
        emdf_onnx_path = args.get("emdf_onnx_path")
        grig_onnx_path = args.get("grig_onnx_path")
        text_detection_onnx_path = args.get("text_detection_onnx_path")
        yolo_detection_onnx_path = args.get("yolo_detection_onnx_path")
        segment_onnx_dir = args.get("segment_onnx_dir")
        refine_type = args.get("refine_type")
        watermark_type = args.get("watermark_type")
        ai_detect_type = args.get("ai_detect_type")
        ai_interactive_type = args.get("ai_interactive_type")
        ai_interactive_prompt = args.get("ai_interactive_prompt")
        ai_interactive_boxes = args.get("ai_interactive_boxes")
        watermark_confidence = args.get("watermark_confidence")
        dilate_num = args.get("dilate_num")
        callback_func = args.get("callback_func", None)
        kwargs = args.get("kwargs", {})
        total_frames = len(frame_mask_map.keys())

        for frame_file, tmp_mask_path in frame_mask_map.items():
            output_frame_path = os.path.join(str(processed_frames_dir), os.path.basename(str(frame_file)))
            ImageWatermarkRemove().run(
                frame_file,
                output_frame_path,
                sr_segment_onnx_path=sr_segment_onnx_path,
                pt_segment_onnx_path=pt_segment_onnx_path,
                pt_inpaint_onnx_path=pt_inpaint_onnx_path, 
                cf_onnx_path=cf_onnx_path, 
                lama_onnx_path=lama_onnx_path,
                emdf_onnx_path=emdf_onnx_path,
                grig_onnx_path=grig_onnx_path,
                text_detection_onnx_path=text_detection_onnx_path,
                yolo_detection_onnx_path=yolo_detection_onnx_path,
                segment_onnx_dir=segment_onnx_dir,
                mask_path=str(tmp_mask_path),
                refine_type=refine_type,
                watermark_type=watermark_type,
                ai_detect_type=ai_detect_type,
                ai_interactive_type=ai_interactive_type,
                ai_interactive_prompt=ai_interactive_prompt,
                ai_interactive_boxes=ai_interactive_boxes,
                watermark_confidence=watermark_confidence,
                dilate_num=dilate_num,
                **kwargs
            )
            if callback_func:
                callback_func(len(os.listdir(str(processed_frames_dir))), total_frames)

    def _video_model_inpainting(self, args: dict):
        frame_files = args.get("frame_files")
        processed_frames_dir = args.get("processed_frames_dir")
        frame_mask_map = args.get("frame_mask_map")
        refine_type = args.get("refine_type")
        dilate_num = args.get("dilate_num")
        if refine_type == "ppt":
            ppt_onnx_basedir = args.get("ppt_onnx_basedir")
            ONNX_PATHS = {
                "raft": os.path.join(ppt_onnx_basedir, "raft", "raft_iter20.encmodel"),
                "recurrent_flow_complete": os.path.join(ppt_onnx_basedir, "recurrent_flow_completion"),
                "ppt": {
                    'encoder': os.path.join(ppt_onnx_basedir, "propagation_transformer", "encoder.encmodel"),
                    'decoder': os.path.join(ppt_onnx_basedir, "propagation_transformer", "decoder.encmodel"),
                    "image_prop_step": os.path.join(ppt_onnx_basedir, "propagation_transformer", "img_prop_step.encmodel"),
                    'ss': os.path.join(ppt_onnx_basedir, "propagation_transformer", "soft_split.encmodel"),
                    'sc': os.path.join(ppt_onnx_basedir, "propagation_transformer", "soft_comp.encmodel"),
                    'bp_backward_step': os.path.join(ppt_onnx_basedir, "propagation_transformer", "backward_step.encmodel"),
                    'bp_forward_step': os.path.join(ppt_onnx_basedir, "propagation_transformer", "forward_step.encmodel"),
                    'bp_backward_first': os.path.join(ppt_onnx_basedir, "propagation_transformer", "backward_first.encmodel"),
                    'bp_forward_first': os.path.join(ppt_onnx_basedir, "propagation_transformer", "forward_first.encmodel"),
                    'bp_fusion': os.path.join(ppt_onnx_basedir, "propagation_transformer", "fusion.encmodel"),
                    'transformer': {
                        'core': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"attention_{i}_core.encmodel") for i in range(8)],
                        'attn': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"attention_{i}_comp.encmodel") for i in range(8)],
                        'proj': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"output_{i}_proj.encmodel") for i in range(8)],
                        'norm1': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"norm1_{i}_comp.encmodel") for i in range(8)],
                        'norm2': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"norm2_{i}_comp.encmodel") for i in range(8)],
                        'mlp': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"mlp_{i}_comp.encmodel") for i in range(8)],
                    }
                }
            }
            if len(frame_mask_map.keys()) < len(frame_files):
                input_frames_dir = os.path.dirname(str(frame_files[0]))
                masks_dir = os.path.dirname(str(list(frame_mask_map.values())[0]))
                input_frames_dir = os.path.join(os.path.dirname(input_frames_dir), "real_{0}_tmp".format(os.path.basename(input_frames_dir)))
                masks_dir = os.path.join(os.path.dirname(masks_dir), "real_{0}_tmp".format(os.path.basename(masks_dir)))
                os.makedirs(input_frames_dir, exist_ok=True)
                os.makedirs(masks_dir, exist_ok=True)
                for frame_file, mask_file in frame_mask_map.items():
                    shutil.copy2(str(frame_file), os.path.join(input_frames_dir, os.path.basename(str(frame_file))))
                    shutil.copy2(str(mask_file), os.path.join(masks_dir, os.path.basename(str(mask_file))))
            else:
                input_frames_dir = os.path.dirname(str(frame_files[0]))
                masks_dir = os.path.dirname(frame_mask_map[str(frame_files[0])])
            output_dir = str(processed_frames_dir)
            cropped_out_dir = os.path.join(os.path.dirname(output_dir), "{0}_cropped".format(os.path.basename(output_dir)))
            bbox = self._get_roi(masks_dir)
            new_input_frames_dir, new_masks_dir = self._crop_frames_and_masks(input_frames_dir, masks_dir, bbox)
            xmin, ymin, xmax, ymax = bbox
            process_w, process_h = xmax - xmin, ymax - ymin
            if max(process_h, process_w) < 540:
                resize_ratio = 1.0
                subvideo_length = 80
            elif max(process_h, process_w) < 720:
                resize_ratio = 540.0 / max(process_h, process_w)
                subvideo_length = 70
            else:
                resize_ratio = 720.0 / max(process_h, process_w)
                subvideo_length = 60
            pipeline = PPTInferenceORT(
                onnx_paths=ONNX_PATHS,
                resize_ratio=resize_ratio,
                height=-1,
                width=-1,
                mask_dilation=dilate_num+2,
                ref_stride=10,
                neighbor_length=10,
                subvideo_length=subvideo_length
            )
            pipeline.inference(input_frames_dir=new_input_frames_dir, masks_dir=new_masks_dir, output_dir=cropped_out_dir)
            pipeline.release()
            del pipeline
            self._past_frames_back(input_frames_dir, cropped_out_dir, output_dir, bbox)
        else:
            raise ValueError(f"Unsupported video inpainting model: {refine_type}")

    def _merge_video_prepare(self, all_frame_files: list, output_dir):
        processed_frames_name = os.listdir(output_dir)
        for one_frame_file in all_frame_files:
            if one_frame_file.name in processed_frames_name:
                continue
            shutil.copy2(str(one_frame_file), os.path.join(output_dir, one_frame_file.name))

    def process_video(
            self, 
            input_video_path, 
            output_video_path,
            sr_segment_onnx_path,
            pt_segment_onnx_path,
            pt_inpaint_onnx_path, 
            cf_onnx_path, 
            lama_onnx_path,
            emdf_onnx_path,
            grig_onnx_path,
            text_detection_onnx_path,
            yolo_detection_onnx_path,
            segment_onnx_dir,
            ppt_onnx_basedir,
            tacker_onnx_dir,
            mask_path: str = "",
            refine_type: str = "coordfill",
            use_cache_mask: bool = False,
            watermark_type: str = "all",
            ai_detect_type: str = "ai_interactive_detect",      # ai_interactive_detect/ai_auto_detect
            ai_interactive_type: str = "semantic_detect",       # semantic_detect/space_detect
            ai_interactive_prompt: str = "watermark",
            ai_interactive_boxes: list = [],
            watermark_confidence: float = 0.5,
            watermark_boxes: list = [],
            watermark_tracking_data: dict = {},
            dilate_num: int = 2,
            ffmpeg_path: str = "",
            callback_func = None,
            **kwargs
        ):
        progress_cb = kwargs.pop("progress_cb", None)
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            ffmpeg_path = os.path.dirname(ffmpeg_path)
        os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']

        probe = ffmpeg.probe(input_video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if not video_stream:
            raise ValueError(f"Can not get {input_video_path} stream information.")
        
        fps = float(video_stream["avg_frame_rate"].split("/")[0]) / float(video_stream["avg_frame_rate"].split("/")[1])
        
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

        if watermark_tracking_data and watermark_tracking_data.get("keyframes") and watermark_tracking_data.get("end_frame") is not None:
            self.video_tracking_data = watermark_tracking_data
        else:
            self.video_tracking_data = None
        self.watermark_boxes = watermark_boxes
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frames_dir = temp_path / 'frames'
            frames_dir.mkdir()
            (
                ffmpeg
                .input(input_video_path)
                .output(str(frames_dir / '%06d.png'), start_number=0, fps_mode="passthrough")
                .overwrite_output()
                .global_args("-hide_banner", "-loglevel", "error")
                .run(capture_stdout=True, capture_stderr=True)
            )
            frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix == '.png'])
            processed_frames_dir = temp_path / 'processed_frames'
            processed_frames_dir.mkdir()
            tmp_mask_dir = temp_path / 'masks'
            tmp_mask_dir.mkdir()
            if progress_cb is not None:
                progress_cb("MaskStart", "")
            frame_mask_map = {}
            self._prepare_masks(
                tmp_mask_dir=tmp_mask_dir, 
                frame_mask_map=frame_mask_map, 
                mask_path=mask_path, 
                use_cache_mask=use_cache_mask, 
                watermark_type=watermark_type,
                ai_detect_type=ai_detect_type,
                ai_interactive_type=ai_interactive_type,
                ai_interactive_prompt=ai_interactive_prompt,
                ai_interactive_boxes=ai_interactive_boxes,
                watermark_confidence=watermark_confidence,
                frame_files=frame_files,
                sr_segment_onnx_path=sr_segment_onnx_path,
                pt_segment_onnx_path=pt_segment_onnx_path,
                text_detection_onnx_path=text_detection_onnx_path,
                yolo_detection_onnx_path=yolo_detection_onnx_path,
                segment_onnx_dir=segment_onnx_dir,
                tacker_onnx_dir=tacker_onnx_dir,
                **kwargs
            )
            if progress_cb is not None:
                tmp_visualzation_path = os.path.join(os.path.dirname(str(tmp_mask_dir)), "masks_visualization")
                for frame_file, tmp_mask_path in frame_mask_map.items():
                    os.makedirs(tmp_visualzation_path, exist_ok=True)
                    self._save_mask_visualization(
                        img_path=str(frame_file),
                        mask=cv2.imread(tmp_mask_path, cv2.IMREAD_GRAYSCALE),
                        file_path=os.path.join(tmp_visualzation_path, os.path.basename(str(frame_file)))
                    )
                output_video_tmp_path = "{0}_mask_visualization.mp4".format(output_video_path.rsplit(".", 1)[0])
                self._merge_video_prepare(frame_files, tmp_visualzation_path)
                self._merge_processed_frames(
                    processed_frames_dir=Path(tmp_visualzation_path),
                    has_audio=has_audio, 
                    fps=fps,
                    input_video_path=input_video_path,
                    output_video_path=output_video_tmp_path
                )
                progress_cb("MaskCompleted", output_video_tmp_path)
            
            if refine_type not in self.video_models_name:
                args = {
                    "processed_frames_dir": processed_frames_dir,
                    "frame_mask_map": frame_mask_map,
                    "sr_segment_onnx_path": sr_segment_onnx_path,
                    "pt_segment_onnx_path": pt_segment_onnx_path,
                    "pt_inpaint_onnx_path": pt_inpaint_onnx_path,
                    "cf_onnx_path": cf_onnx_path,
                    "lama_onnx_path": lama_onnx_path,
                    "emdf_onnx_path": emdf_onnx_path,
                    "grig_onnx_path": grig_onnx_path,
                    "text_detection_onnx_path": text_detection_onnx_path,
                    "yolo_detection_onnx_path": yolo_detection_onnx_path,
                    "segment_onnx_dir": segment_onnx_dir,
                    "refine_type": refine_type,
                    "watermark_type": watermark_type,
                    "ai_detect_type": ai_detect_type,
                    "ai_interactive_type": ai_interactive_type,
                    "ai_interactive_prompt": ai_interactive_prompt,
                    "ai_interactive_boxes": ai_interactive_boxes,
                    "watermark_confidence": watermark_confidence,
                    "dilate_num": dilate_num,
                    "callback_func": callback_func,
                    "kwargs": kwargs
                }
                self._image_model_inpainting(args)
            else:
                args = {
                    "frame_files": frame_files,
                    "processed_frames_dir": processed_frames_dir,
                    "frame_mask_map": frame_mask_map,
                    "dilate_num": dilate_num,
                    "refine_type": refine_type,
                    "ppt_onnx_basedir": ppt_onnx_basedir
                }
                self._video_model_inpainting(args)
            if progress_cb is not None:
                progress_cb("WaterRemoved", "")
            self._merge_video_prepare(frame_files, processed_frames_dir)
            self._merge_processed_frames(
                processed_frames_dir=processed_frames_dir,
                has_audio=has_audio, 
                fps=fps,
                input_video_path=input_video_path,
                output_video_path=output_video_path
            )