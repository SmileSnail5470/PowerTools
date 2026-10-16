import os
import shutil
import app.utils.ffmpeg as ffmpeg
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoInfo:
    fps: float
    has_audio: bool


@dataclass(frozen=True, slots=True)
class VideoFrames:
    directory: Path
    files: tuple[Path, ...]


def _ffmpeg_commands(ffmpeg_path: str) -> tuple[str, str]:
    if not ffmpeg_path:
        return "ffmpeg", "ffprobe"
    configured_path = Path(ffmpeg_path)
    if configured_path.is_dir():
        ffmpeg_path = str(configured_path / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"))
    ffmpeg_file = Path(ffmpeg_path)
    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    return str(ffmpeg_file), str(ffmpeg_file.with_name(ffprobe_name))


def get_video_info(ffmpeg_path: str, input_video_path: str) -> VideoInfo:
    _, ffprobe_cmd = _ffmpeg_commands(ffmpeg_path)
    probe = ffmpeg.probe(input_video_path, cmd=ffprobe_cmd)
    streams = probe["streams"]
    video_stream = next((stream for stream in streams if stream["codec_type"] == "video"), None)
    if video_stream is None:
        raise ValueError(f"Can not get {input_video_path} stream information.")
    fps = float(video_stream["avg_frame_rate"].split("/")[0]) / float(video_stream["avg_frame_rate"].split("/")[1])
    return VideoInfo(fps=fps, has_audio=any(stream["codec_type"] == "audio" for stream in streams))


def extract_video_frames(input_video_path: str, frames_dir: Path, ffmpeg_path: str = "") -> VideoFrames:
    ffmpeg_cmd, _ = _ffmpeg_commands(ffmpeg_path)
    frames_dir.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(input_video_path)
        .output(str(frames_dir / "%06d.png"), start_number=0, fps_mode="passthrough")
        .overwrite_output()
        .global_args("-hide_banner", "-loglevel", "error")
        .run(cmd=ffmpeg_cmd, capture_stdout=True, capture_stderr=True)
    )
    files = tuple(sorted(path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"))
    return VideoFrames(directory=frames_dir, files=files)


def merge_processed_frames(
    frame_files: tuple[Path, ...] | list[Path],
    processed_frames_dir: Path,
    has_audio: bool,
    fps: float,
    input_video_path: str,
    output_video_path: str,
    ffmpeg_path: str = "",
) -> None:
    ffmpeg_cmd, _ = _ffmpeg_commands(ffmpeg_path)
    processed_frames_dir.mkdir(parents=True, exist_ok=True)
    existing_names = {path.name for path in processed_frames_dir.iterdir() if path.is_file()}
    for frame_file in frame_files:
        if frame_file.name in existing_names:
            continue
        shutil.copy2(frame_file, processed_frames_dir / frame_file.name)
    video_input = ffmpeg.input(
        str(processed_frames_dir / "%06d.png"),
        pixel_format="rgb24",
        framerate=fps,
    )
    output_kwargs = {"vcodec": "libx264", "pix_fmt": "yuv420p", "start_number": 0}
    if has_audio:
        audio_input = ffmpeg.input(input_video_path).audio
        stream = ffmpeg.output(video_input, audio_input, output_video_path, **output_kwargs)
    else:
        stream = ffmpeg.output(video_input, output_video_path, **output_kwargs)
    ffmpeg.run(
        stream.global_args("-hide_banner", "-loglevel", "error"),
        cmd=ffmpeg_cmd,
        overwrite_output=True,
        quiet=False,
    )