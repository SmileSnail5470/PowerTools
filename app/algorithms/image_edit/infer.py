import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image
from color_fix import wavelet_color_fix
from pipeline_flux2_klein_onnx import Flux2KleinOnnxPipeline

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class FluxKleinOnnxInference:
    """Drop-in, torch-free replacement for `infer_klein.FluxKleinInference`."""

    def __init__(
        self,
        model_dir: str = "onnx/models",
        device: str = "auto",
        device_id: int = 0,
        low_memory: bool = True,
        use_io_binding: bool | None = None,
        use_cupy: bool | None = None,
        intra_op_num_threads: int | None = None,
        num_inference_steps: int = 4,
        verbose: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.pipe = Flux2KleinOnnxPipeline.from_pretrained(
            model_dir,
            device=device,
            device_id=device_id,
            low_memory=low_memory,
            use_io_binding=use_io_binding,
            use_cupy=use_cupy,
            intra_op_num_threads=intra_op_num_threads,
            verbose=verbose,
        )

    # identical to infer_klein.FluxKleinInference._update_dimensions_from_image
    def _update_dimensions_from_image(self, image_list):
        if image_list is None or len(image_list) == 0:
            return 1024, 1024
        img = image_list[0]
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        if aspect_ratio >= 1:
            new_width = 1024
            new_height = int(1024 / aspect_ratio)
        else:
            new_height = 1024
            new_width = int(1024 * aspect_ratio)
        new_width = round(new_width / 8) * 8
        new_height = round(new_height / 8) * 8
        new_width = max(256, min(1024, new_width))
        new_height = max(256, min(1024, new_height))
        return new_width, new_height

    def _infer(
        self,
        prompt,
        input_images=None,
        seed=42,
        width=1024,
        height=1024,
        num_inference_steps=None,
        guidance_scale=1.0,
    ):
        if guidance_scale > 1.0:
            print(f"[warn] guidance_scale={guidance_scale} is ignored for the distilled model")
        output = self.pipe(
            prompt=prompt,
            image=list(input_images) if input_images else None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            seed=seed,
        )
        self.last_timings = output.timings
        return output.images[0]

    def infer(self, input_path, prompt, output_path, seed: int = 42):
        image_list = [Image.open(p).convert("RGB") for p in self._collect(input_path)]
        width, height = self._update_dimensions_from_image(image_list)
        image = self._infer(
            prompt=prompt, input_images=image_list, width=width, height=height, seed=seed
        )
        image = wavelet_color_fix(image, image_list[0])
        print("complete color fix")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path

    @staticmethod
    def _collect(input_path) -> list[str]:
        path = Path(input_path)
        if path.is_dir():
            return sorted(
                str(p) for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )
        return [str(path)]


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=str(here / "models"), help="exported ONNX bundle")
    p.add_argument("--input", default=str(root / "input" / "000005.png"), help="image file or directory")
    p.add_argument("--prompt", default="Remove Watermark")
    p.add_argument("--output", default=str(root / "output_onnx.png"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "tensorrt", "rocm", "dml", "coreml"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=None, help="override the derived output width")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--threads", type=int, default=None, help="intra_op_num_threads (CPU)")
    p.add_argument(
        "--keep-loaded",
        action="store_true",
        help="keep every session resident (faster for repeated calls, more memory)",
    )
    p.add_argument("--no-io-binding", action="store_true")
    p.add_argument("--no-cupy", action="store_true")
    p.add_argument("--text-only", action="store_true", help="ignore --input, pure text to image")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runner = FluxKleinOnnxInference(
        model_dir=args.model_dir,
        device=args.device,
        device_id=args.device_id,
        low_memory=not args.keep_loaded,
        use_io_binding=False if args.no_io_binding else None,
        use_cupy=False if args.no_cupy else None,
        intra_op_num_threads=args.threads,
        num_inference_steps=args.steps,
    )

    start = time.time()
    if args.text_only:
        image = runner._infer(
            prompt=args.prompt,
            input_images=None,
            seed=args.seed,
            width=args.width or 1024,
            height=args.height or 1024,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
    elif args.width or args.height:
        images = [Image.open(p).convert("RGB") for p in runner._collect(args.input)]
        derived_w, derived_h = runner._update_dimensions_from_image(images)
        image = runner._infer(
            prompt=args.prompt,
            input_images=images,
            seed=args.seed,
            width=args.width or derived_w,
            height=args.height or derived_h,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
    else:
        runner.infer(input_path=args.input, prompt=args.prompt, output_path=args.output, seed=args.seed)

    print(f"Infer cost: {time.time() - start:.2f}s -> {args.output}")
    timings = getattr(runner, "last_timings", None)
    if timings:
        print("  " + "  ".join(f"{k}={v:.2f}s" for k, v in timings.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
