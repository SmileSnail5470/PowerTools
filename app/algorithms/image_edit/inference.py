from pathlib import Path
from PIL import Image
from color_fix import wavelet_color_fix
from app.algorithms.image_edit.pipeline import Pipeline

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class ImageEditInference:
    def __init__(
        self,
        model_dir,
        low_memory: bool = True,
        use_io_binding: bool | None = None,
        use_cupy: bool | None = None,
        intra_op_num_threads: int | None = None,
        num_inference_steps: int = 4,
        verbose: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.pipe = Pipeline.from_pretrained(
            model_dir,
            low_memory=low_memory,
            use_io_binding=use_io_binding,
            use_cupy=use_cupy,
            intra_op_num_threads=intra_op_num_threads,
            verbose=verbose,
        )

    @staticmethod
    def _collect(input_path) -> list[str]:
        path = Path(input_path)
        if path.is_dir():
            return sorted(str(p) for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        return [str(path)]

    def _update_dimensions_from_image(self, image_list):
        if image_list is None or len(image_list) == 0:
            return 1024, 1024, None, None
        img = image_list[0]
        img_width, img_height = img.size
        if max(img_width, img_height) <= 1024:
            new_height = img_height
            new_width = img_width
        else:
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
        return new_width, new_height, img_width, img_height

    def _infer(self, prompt, input_images=None, seed=42, width=1024, height=1024, num_inference_steps=None):
        """
            input_images = None 表示文生图模式
        """
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

    def infer(self, prompt, output_path, input_path = None, mask = None):
        if input_path is not None:
            image_list = [Image.open(p).convert("RGB") for p in self._collect(input_path)]
        else:
            image_list = None
        width, height, ori_width, ori_height = self._update_dimensions_from_image(image_list)
        image = self._infer(prompt=prompt, input_images=image_list, width=width, height=height)
        # image = wavelet_color_fix(image, image_list[0])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if ori_height and ori_width:
            image = image.resize((ori_width, ori_height), resample=Image.Resampling.LANCZOS)
        image.save(output_path)