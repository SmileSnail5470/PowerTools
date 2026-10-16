import os
import cv2
import shutil
import tempfile
import numpy as np
from pathlib import Path
from app.algorithms.private.video_engine.config import EngineConfig
from app.algorithms.private.video_engine.engine import process_video as engine_process_video
from app.algorithms.private.video_engine.plugins.watermark_removal import (
    WatermarkRemovalPlugin,
    WatermarkRemovalPluginConfig,
)
from app.algorithms.visible_watermark_removal.params_config import (
    MaskConfig,
    RestorationRemoveConfig,
    RuntimeConfig,
    VideoEngineRemoveConfig,
    VideoWatermarkRemoveConfig,
)
from app.algorithms.visible_watermark_removal.helpers.video.video_inpainting import (
    image_model_inpainting,
    video_ppt_inpainting,
)
from app.algorithms.visible_watermark_removal.helpers.video.video_io import (
    VideoInfo,
    extract_video_frames,
    get_video_info,
    merge_processed_frames,
)
from app.algorithms.visible_watermark_removal.helpers.video.video_masks import MaskPreparer
from app.algorithms.visible_watermark_removal.helpers.video.video_roi import (
    BBox,
    crop_frames_and_masks,
    expand_bbox_keep_center,
    get_roi,
    paste_cropped_frames,
)
from app.algorithms.visible_watermark_removal.watermark_removal_image import ImageWatermarkRemove


class VideoWatermarkRemover:
    def _expand_bbox_keep_center(self, xmin: int, ymin: int, xmax: int, ymax: int,img_w: int, img_h: int, pad: int = 100) -> BBox:
        return expand_bbox_keep_center(xmin, ymin, xmax, ymax, img_w, img_h, pad)

    def _get_roi(self, masks_dir: str | Path) -> BBox | None:
        return get_roi(Path(masks_dir))

    def _crop_frames_and_masks(self, input_frames_dir: str | Path, masks_dir: str | Path, bbox: BBox | None) -> tuple[Path, Path]:
        return crop_frames_and_masks(Path(input_frames_dir), Path(masks_dir), bbox)

    def _past_frames_back(self, input_frames_dir: str | Path, cropped_out_dir: str | Path, out_path_dir: str | Path, bbox: BBox | None) -> None:
        paste_cropped_frames(Path(input_frames_dir), Path(cropped_out_dir), Path(out_path_dir), bbox)

    def _get_video_info(self, ffmpeg_path: str, input_video_path: str) -> VideoInfo:
        return get_video_info(ffmpeg_path, input_video_path)

    def _extract_input_video_frames(self, input_video_path: str, frames_dir: Path, ffmpeg_path: str = "") -> list[Path]:
        return list(extract_video_frames(input_video_path, frames_dir, ffmpeg_path).files)

    def _save_mask_visualization(self, img_path: str, mask: np.ndarray, file_path: str) -> None:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read frame image: {img_path}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Mask dimensions do not match frame: {img_path}")
        overlay = image.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.55 + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
        if not cv2.imwrite(file_path, overlay):
            raise OSError(f"Failed to write mask visualization: {file_path}")

    def _visualize_mask_video(self, frame_mask_map: dict[str, str], task_dir: Path, output_video_path: str) -> tuple[Path, Path]:
        visualization_dir = task_dir / "masks_visualization"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        for frame_file, mask_path in frame_mask_map.items():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask image: {mask_path}")
            self._save_mask_visualization(frame_file, mask, str(visualization_dir / Path(frame_file).name))
        cache_dir = os.environ.get("POWERTOOLS_TMPE_DIR")
        name_prefix = Path(output_video_path).stem
        if cache_dir:
            output_video_tmp_path = Path(cache_dir) / "tmpdir" / f"{name_prefix}_mask_visualization.mp4"
        else:
            output_video_tmp_path = Path(f"{Path(output_video_path).with_suffix('')}_mask_visualization.mp4")
        output_video_tmp_path.parent.mkdir(parents=True, exist_ok=True)
        return visualization_dir, output_video_tmp_path

    def _merge_video(
        self,
        frame_files: list[Path] | tuple[Path, ...],
        processed_frames_dir: str | Path,
        has_audio: bool,
        fps: float,
        input_video_path: str,
        output_video_path: str,
        ffmpeg_path: str = "",
    ) -> None:
        merge_processed_frames(
            frame_files,
            Path(processed_frames_dir),
            has_audio,
            fps,
            input_video_path,
            output_video_path,
            ffmpeg_path,
        )

    def _notify_mask_visualization(
        self,
        frame_files: list[Path],
        frame_mask_map: dict[str, str],
        task_dir: Path,
        input_video_path: str,
        output_video_path: str,
        info: VideoInfo,
        config: VideoWatermarkRemoveConfig | VideoEngineRemoveConfig,
    ) -> None:
        callback = config.runtime.progress_cb
        if callback is None:
            return
        visualization_dir, visualization_video = self._visualize_mask_video(frame_mask_map, task_dir, output_video_path)
        self._merge_video(
            frame_files,
            visualization_dir,
            info.has_audio,
            info.fps,
            input_video_path,
            str(visualization_video),
            config.runtime.ffmpeg_path,
        )
        callback("MaskCompleted", str(visualization_video))

    def process_video(self, input_video_path: str, output_video_path: str, config: VideoWatermarkRemoveConfig) -> None:
        info = self._get_video_info(config.runtime.ffmpeg_path, input_video_path)
        with tempfile.TemporaryDirectory(prefix="watermark_video_") as temp_dir:
            task_dir = Path(temp_dir)
            frame_files = self._extract_input_video_frames(input_video_path, task_dir / "frames", config.runtime.ffmpeg_path)
            processed_frames_dir = task_dir / "processed_frames"
            processed_frames_dir.mkdir()
            mask_dir = task_dir / "masks"
            progress = config.runtime.progress_cb
            if progress:
                progress("MaskStart", "")
            mask_result = MaskPreparer(config).prepare(mask_dir, frame_files)
            frame_mask_map = mask_result.frame_mask_map
            self._notify_mask_visualization(
                frame_files,
                frame_mask_map,
                task_dir,
                input_video_path,
                output_video_path,
                info,
                config,
            )
            if config.mask.refine_type == "ppt":
                video_ppt_inpainting(
                    frame_files,
                    processed_frames_dir,
                    frame_mask_map,
                    config.mask.dilate_num,
                    config.video_extra.ppt_onnx_basedir,
                    config.runtime.should_cancel,
                )
            else:
                image_model_inpainting(processed_frames_dir, frame_mask_map, config)
            if progress:
                progress("WaterRemoved", "")
            self._merge_video(
                frame_files,
                processed_frames_dir,
                info.has_audio,
                info.fps,
                input_video_path,
                output_video_path,
                config.runtime.ffmpeg_path,
            )

    def process_video_without_mask(self, input_video_path: str, output_video_path: str, config: RestorationRemoveConfig) -> None:
        info = self._get_video_info(config.runtime.ffmpeg_path, input_video_path)
        with tempfile.TemporaryDirectory(prefix="watermark_video_") as temp_dir:
            task_dir = Path(temp_dir)
            frame_files = self._extract_input_video_frames(input_video_path, task_dir / "frames", config.runtime.ffmpeg_path)
            processed_frames_dir = task_dir / "processed_frames"
            processed_frames_dir.mkdir()
            progress = config.runtime.progress_cb
            callback = config.runtime.callback_func
            if progress:
                progress("WaterRemoveStart", "")
            frame_config = RestorationRemoveConfig(
                restoration=config.restoration,
                params=config.params,
                runtime=RuntimeConfig(),
            )
            image_remover = ImageWatermarkRemove()
            for completed, frame_file in enumerate(frame_files, start=1):
                output_path = processed_frames_dir / frame_file.name
                image_remover.run_without_mask(str(frame_file), str(output_path), config=frame_config)
                if callback:
                    callback(completed, len(frame_files))
            if progress:
                progress("WaterRemoved", "")
            self._merge_video(
                frame_files,
                processed_frames_dir,
                info.has_audio,
                info.fps,
                input_video_path,
                output_video_path,
                config.runtime.ffmpeg_path,
            )

    @staticmethod
    def _build_engine(config: VideoEngineRemoveConfig) -> tuple[EngineConfig, WatermarkRemovalPlugin]:
        engine_config = config.engine
        mask_config = config.mask or MaskConfig(refine_type="lama")
        runtime = config.runtime
        engine = EngineConfig(
            prompt=engine_config.prompt,
            task_type=engine_config.task_type,
            rife_model_path=engine_config.model_paths.video_engine_model_dir,
            keyframe_stride=engine_config.keyframe_stride,
            ffmpeg_path=runtime.ffmpeg_path,
            should_cancel=runtime.should_cancel,
            is_static_watermark=engine_config.is_static_watermark,
            progress_cb=runtime.progress_cb,
        )
        plugin_config = WatermarkRemovalPluginConfig(
            segment=config.segment,
            inpaint=config.inpaint,
            detection=config.detection,
            mask=MaskConfig(
                refine_type=mask_config.refine_type,
                dilate_num=mask_config.dilate_num,
            ),
        )
        return engine, WatermarkRemovalPlugin(config=plugin_config)

    @staticmethod
    def _stage_engine_masks(frame_files: list[Path], frame_mask_map: dict[str, str], mask_dir: Path) -> None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        for frame_file in frame_files:
            mask_path = frame_mask_map.get(str(frame_file))
            destination = mask_dir / frame_file.name
            if mask_path is not None:
                shutil.copy2(mask_path, destination)
                continue
            frame = cv2.imread(str(frame_file))
            if frame is None:
                raise ValueError(f"Failed to read frame image: {frame_file}")
            if not cv2.imwrite(str(destination), np.zeros(frame.shape[:2], dtype=np.uint8)):
                raise OSError(f"Failed to write empty frame mask: {destination}")

    def process_video_engine(self, input_video_path: str, output_video_path: str, config: VideoEngineRemoveConfig) -> None:
        engine_config, plugin = self._build_engine(config)
        progress = config.runtime.progress_cb
        engine_settings = config.engine
        if engine_settings.mask_path:
            if progress:
                progress("WaterRemoveStart", "")
            engine_process_video(
                input_video_path,
                output_video_path,
                plugin=plugin,
                config=engine_config,
                mask=engine_settings.mask_path,
            )
            if progress:
                progress("WaterRemoved", "")
        elif not engine_settings.auto_detect_mask:
            if progress:
                progress("WaterRemoveStart", "")
            engine_process_video(
                input_video_path,
                output_video_path,
                plugin=plugin,
                config=engine_config,
                mask=None,
            )
            if progress:
                progress("WaterRemoved", "")
        else:
            if config.mask is None or config.video_extra is None:
                raise ValueError("Automatic mask detection requires video mask and model configurations")
            info = self._get_video_info(config.runtime.ffmpeg_path, input_video_path)
            with tempfile.TemporaryDirectory(prefix="watermark_video_engine_") as temp_dir:
                task_dir = Path(temp_dir)
                frame_files = self._extract_input_video_frames(input_video_path, task_dir / "frames", config.runtime.ffmpeg_path)
                mask_dir = task_dir / "masks"
                if progress:
                    progress("MaskStart", "")
                detection_config = VideoWatermarkRemoveConfig(
                    segment=config.segment,
                    inpaint=config.inpaint,
                    video_extra=config.video_extra,
                    detection=config.detection,
                    mask=config.mask,
                    runtime=config.runtime,
                )
                frame_mask_map = MaskPreparer(detection_config).prepare(mask_dir, frame_files).frame_mask_map
                self._notify_mask_visualization(
                    frame_files,
                    frame_mask_map,
                    task_dir,
                    input_video_path,
                    output_video_path,
                    info,
                    config,
                )
                engine_mask_dir = task_dir / "engine_masks"
                self._stage_engine_masks(frame_files, frame_mask_map, engine_mask_dir)
                engine_process_video(
                    input_video_path,
                    output_video_path,
                    plugin=plugin,
                    config=engine_config,
                    mask=str(engine_mask_dir),
                )
                if progress:
                    progress("WaterRemoved", "")
