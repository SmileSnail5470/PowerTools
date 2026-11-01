import logging
from dataclasses import asdict
from app.workers.work_base import BaseWorker
from app.workers.watermark_remove import VideoTrackingParams

logger = logging.getLogger("VideoObjectDetectorWorker")


class VideoObjectDetectorWorker(BaseWorker):
    def __init__(self, file_path: str, params: VideoTrackingParams, total_frames: int = 0):
        super().__init__(file_path, params, total_frames)

    def run_algorithm(self, progress_cb, cancel_requested, file_path: str, params, total_frames: int):
        tracking_data = asdict(params)
        tracking_data["enable_preview_mode"] = False  # 前台模式实际运行，非预览
        output_path = tracking_data.get("tracking_detect_mask_path")
        logger.info(
            "VideoObjectDetectorWorker: start video=%s algorithm=%s anchor=%s direction=%s range=%s",
            file_path,
            tracking_data.get("tracking_algorithm"),
            tracking_data.get("tracking_detect_anchor_frame"),
            tracking_data.get("tracking_detect_direction"),
            tracking_data.get("tracking_detect_frame_range"),
        )
        if progress_cb:
            progress_cb("SegmentDone", "")

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task cancelled")

        # 跟踪阶段（逐帧推进）
        # 此处为占位实现：实际业务逻辑由上层集成时替换或继承本类重写
        # 当有真实算法时，在此循环内每帧调用 progress_cb(frame_idx, "frame")
        processed = 0
        for frame_idx in range(1, max(total_frames, 1) + 1):
            if cancel_requested and cancel_requested():
                raise InterruptedError("Task cancelled at frame %d" % frame_idx)
            processed = frame_idx
            if progress_cb:
                progress_cb(processed, "frame")

        if progress_cb:
            progress_cb("TrackDone", "")

        logger.info("VideoObjectDetectorWorker: finished, processed=%d frames, output=%s", processed, output_path)
        return output_path
