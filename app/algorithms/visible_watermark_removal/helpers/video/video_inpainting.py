import logging
import os
import shutil
import tempfile
from pathlib import Path
from app.algorithms.visible_watermark_removal.params_config import (
    ImageWatermarkRemoveConfig,
    MaskConfig,
    RuntimeConfig,
    VideoWatermarkRemoveConfig,
    get_ppt_onnx_path,
)
from app.algorithms.visible_watermark_removal.video_modules.ppt.inference import PPTInferenceORT
from app.algorithms.visible_watermark_removal.helpers.video.video_roi import (
    crop_frames_and_masks,
    get_roi,
    paste_cropped_frames,
)
from app.algorithms.visible_watermark_removal.watermark_removal_image import ImageWatermarkRemove


logger = logging.getLogger("subprocess")


def image_model_inpainting(
    processed_frames_dir: Path, 
    frame_mask_map: dict[str, str], 
    config: VideoWatermarkRemoveConfig
) -> None:
    callback = config.runtime.callback_func
    total_frames = len(frame_mask_map.keys())
    image_remover = ImageWatermarkRemove()
    for completed, (frame_file, mask_path) in enumerate(frame_mask_map.items(), start=1):
        output_path = processed_frames_dir / Path(frame_file).name
        frame_config = ImageWatermarkRemoveConfig(
            segment=config.segment,
            inpaint=config.inpaint,
            detection=config.detection,
            mask=MaskConfig(
                mask_path=mask_path,
                refine_type=config.mask.refine_type,
                dilate_num=config.mask.dilate_num,
            ),
            runtime=RuntimeConfig(),
        )
        image_remover.run(frame_file, str(output_path), config=frame_config)
        if callback:
            callback(completed, total_frames)


def _ppt_processing_parameters(width: int, height: int) -> tuple[float, int]:
    initial_subvideo_length = int(os.environ.get("POWERTOOLS_SUBVIDEO_LENGTH", 100))
    gpu_memory_limit = int(os.environ.get("POWERTOOLS_GPU_MEMORY_LIMIT", 16))
    longest_side = max(width, height)
    if longest_side < 540:
        resize_ratio = 1.0
        subvideo_length = 80 if gpu_memory_limit > 12 else 70 if gpu_memory_limit > 8 else 60
    elif longest_side < 720:
        resize_ratio = 540.0 / longest_side
        subvideo_length = 70 if gpu_memory_limit > 12 else 60 if gpu_memory_limit > 8 else 50
    else:
        resize_ratio = 720.0 / longest_side
        subvideo_length = 60 if gpu_memory_limit > 12 else 50 if gpu_memory_limit > 8 else 40
    return resize_ratio, min(subvideo_length, initial_subvideo_length)


def video_ppt_inpainting(
    frame_files: list[Path] | tuple[Path, ...], 
    processed_frames_dir: Path, 
    frame_mask_map: dict[str, str], 
    dilate_num: int, 
    ppt_onnx_basedir: str
) -> None:
    if not frame_mask_map:
        raise ValueError("PPT inpainting requires at least one frame mask")
    with tempfile.TemporaryDirectory(prefix="ppt_frames_") as input_tmp, tempfile.TemporaryDirectory(prefix="ppt_masks_") as masks_tmp:
        input_frames_dir = Path(input_tmp)
        masks_dir = Path(masks_tmp)
        frames_by_path = {str(frame_file): frame_file for frame_file in frame_files}
        for frame_path, mask_path in frame_mask_map.items():
            frame_file = frames_by_path.get(frame_path)
            if frame_file is None:
                raise ValueError(f"Mask mapping refers to an unknown frame: {frame_path}")
            shutil.copy2(frame_file, input_frames_dir / frame_file.name)
            shutil.copy2(mask_path, masks_dir / frame_file.name)
        bbox = get_roi(masks_dir)
        cropped_frames_dir, cropped_masks_dir = crop_frames_and_masks(input_frames_dir, masks_dir, bbox)
        if bbox is None:
            raise Exception(f"Not find mask in input video")
        else:
            xmin, ymin, xmax, ymax = bbox
            width, height = xmax - xmin, ymax - ymin
        resize_ratio, subvideo_length = _ppt_processing_parameters(width, height)
        logger.info(
            "Process video resize ratio %s and subvideo length %s and process size %s "
            "and gpu memory limit %s",
            resize_ratio,
            subvideo_length,
            (width, height),
            os.environ.get("POWERTOOLS_GPU_MEMORY_LIMIT", 16),
        )
        pipeline = PPTInferenceORT(
            onnx_paths=get_ppt_onnx_path(ppt_onnx_basedir),
            resize_ratio=resize_ratio,
            height=-1,
            width=-1,
            mask_dilation=dilate_num + 2,
            ref_stride=10,
            neighbor_length=10,
            subvideo_length=subvideo_length,
        )
        try:
            cropped_output_dir = processed_frames_dir.parent / f"{processed_frames_dir.name}_cropped"
            pipeline.inference(
                input_frames_dir=str(cropped_frames_dir),
                masks_dir=str(cropped_masks_dir),
                output_dir=str(cropped_output_dir),
            )
        finally:
            pipeline.release()
        paste_cropped_frames(
            input_frames_dir,
            cropped_output_dir,
            processed_frames_dir,
            bbox,
        )
