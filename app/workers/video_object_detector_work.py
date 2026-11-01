import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.workers.watermark_remove import VideoTrackingParams
from app.ui.common.config import cfg
from app.algorithms.visible_watermark_removal.helpers.video.video_io import extract_video_frames
from app.algorithms.visible_watermark_removal.helpers.video.video_masks import MaskPreparer
from app.algorithms.visible_watermark_removal.params_config import (
    DetectionConfig,
    RuntimeConfig,
    SegmentModelPaths,
    InpaintModelPaths,
    VideoExtraModelPaths,
    VideoMaskConfig,
    VideoWatermarkRemoveConfig,
)

logger = logging.getLogger("VideoObjectDetectorWorker")


class VideoObjectDetectorWorker(BaseWorker):
    def __init__(self, file_path: str, params: VideoTrackingParams):
        super().__init__(file_path, params)

    def run_algorithm(self, progress_cb, cancel_requested, file_path: str, params: VideoTrackingParams) -> str:
        os.environ["_feature_name_"] = "video_object_detector"
        if "POWERTOOLS_TMPE_DIR" not in os.environ:
            os.environ["POWERTOOLS_TMPE_DIR"] = cfg.get(cfg.cachePath)
        deps_path = cfg.get(cfg.localAIModelDeps)
        hw = _resolve_hardware_variant()

        onnx_model_dir         = os.path.join(deps_path, hw, "visible_watermark_removal")
        segment_model_dir      = os.path.join(deps_path, hw, "object_segmentation")
        general_edit_dir       = os.path.join(deps_path, hw, "image_edit", "general_edit")
        object_remove_dir      = os.path.join(deps_path, hw, "image_edit", "object_remove")
        ppt_onnx_basedir       = os.path.join(deps_path, hw, "video_inpainting", "ppt")
        tacker_onnx_dir        = os.path.join(deps_path, hw, "tracker")

        segment_paths = SegmentModelPaths(
            sr_segment_onnx_path=os.path.join(onnx_model_dir, "sr_segment.encmodel"),
            pt_segment_onnx_path=os.path.join(onnx_model_dir, "pt_segment.encmodel"),
            text_detection_onnx_path=os.path.join(onnx_model_dir, "pp_ocr_det.encmodel"),
            yolo_detection_onnx_path=os.path.join(onnx_model_dir, "yolo.encmodel"),
            segment_onnx_dir=segment_model_dir,
        )
        inpaint_paths = InpaintModelPaths(
            pt_inpaint_onnx_path=os.path.join(onnx_model_dir, "pt_inpaint.encmodel"),
            cf_onnx_path=os.path.join(onnx_model_dir, "cf.encmodel"),
            lama_onnx_path=os.path.join(onnx_model_dir, "lama_fp32.encmodel"),
            emdf_onnx_path=os.path.join(onnx_model_dir, "emdf_inpaint.encmodel"),
            grig_onnx_path=os.path.join(onnx_model_dir, "grig_inpaint.encmodel"),
            general_edit_onnx_dir=general_edit_dir,
            object_remove_onnx_dir=object_remove_dir,
        )
        tracking_data = asdict(params)

        if params.tracking_detect_prompt:
            watermark_ai_detect_method = "semantic_detect"
        elif not params.tracking_detect_prompt and params.tracking_detect_pos_boxes:
            watermark_ai_detect_method = "space_detect"
        else:
            raise ValueError("Either tracking_detect_prompt or tracking_detect_pos_boxes must be provided for detection.")
        detection_cfg = DetectionConfig(
            watermark_ai_detect_method=watermark_ai_detect_method,
            watermark_ai_detect_prompt=params.tracking_detect_prompt,
            watermark_ai_detect_pos_boxes=params.tracking_detect_pos_boxes,
            watermark_ai_detect_neg_boxes=params.tracking_detect_neg_boxes,
            watermark_ai_detect_confidence=params.tracking_detect_confidence,
        )

        runtime_cfg = RuntimeConfig(
            progress_cb=None,
            ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
            should_cancel=cancel_requested,
        )

        if params.tracking_detect_mask_path:
            mask_out_dir = Path(params.tracking_detect_mask_path)
            mask_out_dir.mkdir(parents=True, exist_ok=True)
        else:
            cache_root = cfg.get(cfg.cachePath) or tempfile.gettempdir()
            mask_out_dir = Path(cache_root) / "video_detector_masks" / Path(file_path).stem
            mask_out_dir.mkdir(parents=True, exist_ok=True)
        tracking_data["tracking_detect_mask_path"] = mask_out_dir

        video_cfg = VideoWatermarkRemoveConfig(
            segment=segment_paths,
            inpaint=inpaint_paths,
            video_extra=VideoExtraModelPaths(
                ppt_onnx_basedir=ppt_onnx_basedir,
                tacker_onnx_dir=tacker_onnx_dir,
                object_segmentation_model_dir=segment_model_dir,
            ),
            detection=detection_cfg,
            mask=VideoMaskConfig(
                mask_path="",
                use_cache_mask=False,
                video_watermark_ai_detect_strategy="tracking",
                video_watermark_ai_detect_anchor_frame=params.tracking_detect_anchor_frame,
                video_watermark_ai_detect_frame_range=params.tracking_detect_frame_range,
                watermark_tracking_data=tracking_data,
            ),
            runtime=runtime_cfg,
        )

        logger.info(
            "VideoObjectDetectorWorker: start video=%s algorithm=%s anchor=%s direction=%s range=%s output=%s",
            file_path,
            params.tracking_algorithm,
            params.tracking_detect_anchor_frame,
            params.tracking_detect_direction,
            params.tracking_detect_frame_range,
            mask_out_dir,
        )

        with tempfile.TemporaryDirectory(prefix="video_detector_frames_") as frames_tmp:
            frame_files = list(extract_video_frames(file_path, Path(frames_tmp), runtime_cfg.ffmpeg_path).files)
            if cancel_requested and cancel_requested():
                raise InterruptedError("Task cancelled before mask generation")
            mask_result = MaskPreparer(video_cfg).prepare(mask_out_dir, frame_files)

        logger.info("VideoObjectDetectorWorker: finished, output=%s", mask_out_dir)
        return str(mask_result.mask_dir)
