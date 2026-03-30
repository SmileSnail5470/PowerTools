import os
import numpy as np
import onnxruntime as ort
from app.algorithms.blind_watermark_addition.ecc_utils import HammingECC
import app.utils.ffmpeg as ffmpeg


class VideoBlindWatermarkEmbed():
    def __init__(self):
        pass

    def _create_predictor(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        inputs = self.session.get_inputs()
        self.image_input = inputs[0].name
        self.msg_input  = inputs[1].name

        self.output_name = self.session.get_outputs()[0].name

    def __embed_video_clip(self, clip: np.ndarray, msgs: np.ndarray) -> np.ndarray:
        clip_tensor = np.transpose(clip, (0, 3, 1, 2)).astype(np.float32) / 255.0
        imgs_ws = self.session.run(
            [self.output_name],
            {
                self.image_input: clip_tensor,
                self.msg_input: msgs
            }
        )
        processed_clip = imgs_ws[0]
        return processed_clip

    def prepare(self, onnx_path: str, ffmpeg_path: str):
        self.onnx_path = onnx_path
        self._create_predictor()
        if os.path.isfile(ffmpeg_path):
            ffmpeg_path = os.path.dirname(ffmpeg_path)
        os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']

    def watermark_addition(self, input_path: str, output_file: str, message: str, chunk_size: int = 8) -> None:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_file

        probe = ffmpeg.probe(input_path)
        video_info = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])
        fps = float(video_info["r_frame_rate"].split("/")[0]) / float(video_info["r_frame_rate"].split("/")[1])
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

        # Open the input video
        process1 = (
            ffmpeg.input(input_path)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(width, height),
                r=fps,
            )
            .global_args("-hide_banner", "-loglevel", "error")
            .run_async(pipe_stdout=True)
        )
        # Open the output video with optimal thread usage.
        process2 = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(width, height),
                r=fps,
            )
            .output(output_path, vcodec="libx264", pix_fmt="yuv420p", r=fps)
            .global_args("-hide_banner", "-loglevel", "error")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        ecc = HammingECC()
        if os.getenv("BLIND_WATERMARK_CHARSET") is not None:
            charset = os.getenv("BLIND_WATERMARK_CHARSET")
            ecc.CHARSET = charset
        msgs, _ = ecc.str_to_tensor(message)

        # Process the video
        frame_size = width * height * 3
        chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
        frames_in_chunk = 0

        for in_bytes in iter(lambda: process1.stdout.read(frame_size), b""):
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            chunk[frames_in_chunk] = frame
            frames_in_chunk += 1
            if frames_in_chunk == chunk_size:
                processed_frames = self.__embed_video_clip(chunk, msgs)
                process2.stdin.write(processed_frames.tobytes())
                frames_in_chunk = 0

        if frames_in_chunk > 0:
            out = chunk[:frames_in_chunk]
            pad = np.repeat(out[-1:], 8 - frames_in_chunk, axis=0)
            out = np.concatenate([out, pad], axis=0)
            processed_frames = self.__embed_video_clip(out, msgs)
            processed_frames = processed_frames[:frames_in_chunk]
            process2.stdin.write(processed_frames.tobytes())

        process1.stdout.close()
        process2.stdin.close()
        process1.wait()
        process2.wait()

        # Copy just the audio from the original video
        temp_output = output_path + ".tmp"
        if os.path.exists(temp_output):
            os.remove(temp_output)
        os.rename(output_path, temp_output)

        videostream = ffmpeg.input(temp_output)
        if has_audio:
            audiostream = ffmpeg.input(input_path)
            process3 = (
                ffmpeg.output(
                    videostream,
                    audiostream,
                    output_path,
                    vcodec="copy",
                    acodec="copy",
                )
                .global_args("-hide_banner", "-loglevel", "error")
                .overwrite_output()
                .run_async()
            )
        else:
            process3 = (
                ffmpeg.output(
                    videostream,
                    output_path,
                    vcodec="copy",
                )
                .global_args("-hide_banner", "-loglevel", "error")
                .overwrite_output()
                .run_async()
            )
        process3.wait()
        os.remove(temp_output)


class VideoBlindWatermarkDetect():
    def __init__(self):
        pass

    def _create_predictor(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        inputs = self.session.get_inputs()
        self.image_input = inputs[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __detect_video_clip(self, clip: np.ndarray):
        clip_tensor = np.transpose(clip, (0, 3, 1, 2)).astype(np.float32) / 255.0
        output_bits = self.session.run(
            [self.output_name],
            {
                self.image_input: clip_tensor
            }
        )[0]
        return output_bits

    def prepare(self, onnx_path: str, ffmpeg_path: str):
        self.onnx_path = onnx_path
        self._create_predictor()
        if os.path.isfile(ffmpeg_path):
            ffmpeg_path = os.path.dirname(ffmpeg_path)
        os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']

    def watermark_extraction(self, input_path: str, chunk_size: int = 8) -> None:   
        probe = ffmpeg.probe(input_path)
        video_info = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])
        num_frames = int(probe["streams"][0]["nb_frames"])

        # Open the input video
        process1 = (
            ffmpeg.input(input_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .global_args("-hide_banner", "-loglevel", "error")
            .run_async(pipe_stdout=True)
        )

        # Process the video
        frame_size = width * height * 3
        chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
        frame_count = 0
        soft_msgs = []
        while True:
            in_bytes = process1.stdout.read(frame_size)
            if not in_bytes:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            chunk[frame_count % chunk_size] = frame
            frame_count += 1
            if frame_count % chunk_size == 0:
                soft_msgs.append(self.__detect_video_clip(chunk))
        process1.stdout.close()
        process1.wait()

        soft_msgs = np.concatenate(soft_msgs, axis=0)
        soft_msgs = soft_msgs.mean(axis=0)
        ecc = HammingECC()
        if os.getenv("BLIND_WATERMARK_CHARSET") is not None:
            charset = os.getenv("BLIND_WATERMARK_CHARSET")
            ecc.CHARSET = charset
        preds_str, _ = ecc.tensor_to_string(soft_msgs)
        metrics = {
            "file": input_path,
            "preds": preds_str
        }
        return metrics
    

if __name__ == "__main__":
    ffmpeg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffmpeg-linux64", "bin")
    embed_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnxmodel", "pixelseal_video_embed.onnx")
    detect_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnxmodel", "pixelseal_video_detect.onnx")
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "videos", "1.mp4")
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "1.mp4")

    embed_instance = VideoBlindWatermarkEmbed()
    embed_instance.prepare(onnx_path=embed_onnx_path, ffmpeg_path=ffmpeg_path)
    embed_instance.watermark_addition(input_path=image_path, output_file=output_path, message="HELLO,1234")
    print(f"embed {output_path} success.")

    detect_instance = VideoBlindWatermarkDetect()
    detect_instance.prepare(onnx_path=detect_onnx_path, ffmpeg_path=ffmpeg_path)
    res = detect_instance.watermark_extraction(input_path=output_path)
    print(res["preds"])