import os
import platform
import sys
import cv2
import numpy as np
import onnxruntime as ort
ort.preload_dlls(directory="")
from app.algorithms import general_inference_session, general_session, general_provider, ORTEnvironment
ORTEnvironment.initialize()


class PatchWiperSegment():
    def __init__(self):
        self.patch_size = 512
        self.overlap = 0

    def _create_predictor(self):
        session_options = general_session()
        providers, provider_options = general_provider()
        self.session = general_inference_session(
            self.onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def prepare(self, onnx_path: str = None):
        self.onnx_path = onnx_path
        self._create_predictor()

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0    # [H,W,C]
        img = np.transpose(img, (2, 0, 1))      # CHW
        img = img[np.newaxis, ...]              # [1,C,H,W]

        return img
    
    def _pad_to_multiple(self, img: np.ndarray, multiple=32):
        """
        img: [1, C, H, W]
        return: padded_img, (pad_bottom, pad_right)
        """
        _, _, H, W = img.shape

        new_H = ((H + multiple - 1) // multiple) * multiple
        new_W = ((W + multiple - 1) // multiple) * multiple

        pad_bottom = new_H - H
        pad_right = new_W - W

        padded = np.pad(
            img,
            ((0, 0), (0, 0), (0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=0
        )

        return padded, (pad_bottom, pad_right)
    
    def _unpad(self, img: np.ndarray, pads):
        pad_bottom, pad_right = pads
        _, _, H, W = img.shape

        h_end = H - pad_bottom if pad_bottom > 0 else H
        w_end = W - pad_right if pad_right > 0 else W

        return img[:, :, :h_end, :w_end]
    
    def _process_img(self, image_path: str):
        img = self._load_image(image_path)
        img, pads = self._pad_to_multiple(img)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: img}
        )
        mask = outputs[0]
        mask = self._unpad(mask, pads)
        return mask
    
    def watermark_segment(self, image_path: str):
        result = {}

        process_ret = self._process_img(image_path=image_path)

        result["mask"] = process_ret

        return result
    

if __name__ == "__main__":
    import os
    from PIL import Image
    input_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "image.jpg")
    detection = PatchWiperSegment()
    detection.prepare(onnx_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "OnnxModels", "pt_segment.encmodel"))

    result = detection.watermark_segment(input_image)

    mask = result["mask"][0, 0]              # [H, W]
    mask = np.clip(mask, 0.0, 1.0)
    mask = (mask * 255.0).astype(np.uint8)

    Image.fromarray(mask).save("mask_output.png")