import app.utils.ffmpeg as ffmpeg
import shutil
import tempfile
import os
from PIL import Image
from pathlib import Path
from app.algorithms.visible_watermark_removal.watermark_removal_image import ImageWatermarkRemove, WatermarkSegment


class VideoWatermarkRemover:
    def __init__(self):
        pass

    def _prepare_masks(
            self, 
            tmp_mask_dir: str|Path, 
            frame_mask_map: dict, 
            mask_path: str, 
            use_cache_mask: bool, 
            watermark_type: str,
            frame_files: list,
            sr_segment_onnx_path,
            pt_segment_onnx_path,
            text_detection_onnx_path,
            yolo_detection_onnx_path,
            **kwargs
        ):
        if use_cache_mask and not mask_path:
            # 静态水印，AI检测
            mask = WatermarkSegment(watermark_type).segment(
                image_path=str(frame_files[0]),
                sr_onnx_path=sr_segment_onnx_path,
                pt_onnx_path=pt_segment_onnx_path,
                text_detection_onnx_path=text_detection_onnx_path,
                yolo_detection_onnx_path=yolo_detection_onnx_path,
                **kwargs
            )
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
            for i, frame_file in enumerate(frame_files):
                mask = WatermarkSegment(watermark_type).segment(
                    image_path=str(frame_file),
                    sr_onnx_path=sr_segment_onnx_path,
                    pt_onnx_path=pt_segment_onnx_path,
                    text_detection_onnx_path=text_detection_onnx_path,
                    yolo_detection_onnx_path=yolo_detection_onnx_path,
                    **kwargs
                )
                tmp_mask_path = os.path.join(str(tmp_mask_dir), frame_file.name)
                Image.fromarray(mask).convert("L").save(tmp_mask_path)
                frame_mask_map[str(frame_file)] = tmp_mask_path

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
            )
        else:
            stream = ffmpeg.output(
                video_input,
                output_video_path,
                **output_kwargs
            )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    
    def process_video(
            self, 
            input_video_path, 
            output_video_path,
            sr_segment_onnx_path,
            pt_segment_onnx_path,
            pt_inpaint_onnx_path, 
            cf_onnx_path, 
            lama_onnx_path,
            text_detection_onnx_path,
            yolo_detection_onnx_path,
            mask_path: str = "",
            image_refine_type: str = "coordfill",  # patchwiper/lama/transparent/cv2/coordfill
            use_cache_mask: bool = False,
            watermark_type: str = "all",      # text / all
            ffmpeg_path: str = "",
            callback_func = None,
            **kwargs
        ):
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            ffmpeg_path = os.path.dirname(ffmpeg_path)
        os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']

        probe = ffmpeg.probe(input_video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if not video_stream:
            raise ValueError(f"Can not get {input_video_path} stream information.")
        
        fps = float(video_stream["avg_frame_rate"].split("/")[0]) / float(video_stream["avg_frame_rate"].split("/")[1])
        
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frames_dir = temp_path / 'frames'
            frames_dir.mkdir()
            
            # 提取所有帧
            (
                ffmpeg
                .input(input_video_path)
                .output(str(frames_dir / '%06d.png'), start_number=0, vsync="passthrough")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # 获取所有帧文件
            frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix == '.png'])
            total_frames = len(frame_files)
            
            # 处理每一帧
            processed_frames_dir = temp_path / 'processed_frames'
            processed_frames_dir.mkdir()
            tmp_mask_dir = temp_path / 'masks'
            tmp_mask_dir.mkdir()

            frame_mask_map = {}

            self._prepare_masks(
                tmp_mask_dir=tmp_mask_dir, 
                frame_mask_map=frame_mask_map, 
                mask_path=mask_path, 
                use_cache_mask=use_cache_mask, 
                watermark_type=watermark_type,
                frame_files=frame_files,
                sr_segment_onnx_path=sr_segment_onnx_path,
                pt_segment_onnx_path=pt_segment_onnx_path,
                text_detection_onnx_path=text_detection_onnx_path,
                yolo_detection_onnx_path=yolo_detection_onnx_path,
                **kwargs
            )
            
            for _, frame_file in enumerate(frame_files):
                output_frame_path = processed_frames_dir / frame_file.name
                ImageWatermarkRemove().run(
                    frame_file,
                    output_frame_path,
                    sr_segment_onnx_path=sr_segment_onnx_path,
                    pt_segment_onnx_path=pt_segment_onnx_path,
                    pt_inpaint_onnx_path=pt_inpaint_onnx_path, 
                    cf_onnx_path=cf_onnx_path, 
                    lama_onnx_path=lama_onnx_path,
                    text_detection_onnx_path=text_detection_onnx_path,
                    yolo_detection_onnx_path=yolo_detection_onnx_path,
                    mask_path=frame_mask_map[str(frame_file)],
                    refine_type=image_refine_type,
                    watermark_type=watermark_type,
                    **kwargs
                )
                if callback_func:
                    callback_func(len(os.listdir(str(processed_frames_dir))), total_frames)

            self._merge_processed_frames(
                processed_frames_dir=processed_frames_dir,
                has_audio=has_audio, 
                fps=fps,
                input_video_path=input_video_path,
                output_video_path=output_video_path
            )


def callback(current_frame, total_frames):
    print(f"处理进度: {current_frame}/{total_frames}")


if __name__ == "__main__":
    refine_type = "coordfill"
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assess", "test.mp4")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"output_{refine_type}.mp4")
    water_remove = VideoWatermarkRemover()
    water_remove.process_video(
        input_video_path=input_path, 
        output_video_path=out_path,
        # mask_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assess", "1_mask.png"),
        sr_segment_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "sr_segment.onnx"),
        pt_segment_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "pt_segment.onnx"),
        pt_inpaint_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "pt_inpaint.onnx"), 
        cf_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "cf.onnx"), 
        lama_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "lama_fp32.onnx"),
        text_detection_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "pp_ocr_det.onnx"),
        yolo_detection_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "yolo.onnx"),
        watermark_type="text",
        use_cache_mask=True,
        image_refine_type=refine_type,
        ffmpeg_path="",
        callback_func=callback,
        text_det_unclip_ratio=1.8
    )