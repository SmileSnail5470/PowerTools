import os
import cv2
import numpy as np
from app.algorithms import general_inference_session, general_provider, general_session, ORTEnvironment
ORTEnvironment.initialize()


class SLBRSegment():
    def __init__(self):
        self.patch_size = 256
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

    def _preprocess(self, file_path: str) -> np.ndarray:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0     # [H,W,C]
        img = np.transpose(img, (2, 0, 1))       # CHW
        img = img[np.newaxis, ...]               # [1,C,H,W]

        return img
    
    def _inference_adaptive(
        self,
        image: np.ndarray,
        batch_size: int = 1
    ):
        B, C, H, W = image.shape
        stride = self.patch_size - self.overlap

        mask_output = np.zeros((B, 1, H, W), dtype=np.float32)
        img_output  = np.zeros((B, C, H, W), dtype=np.float32)
        counter     = np.zeros((B, 1, H, W), dtype=np.float32)

        patches = []
        coords = []

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1, y2 = y, min(y + self.patch_size, H)
                x1, x2 = x, min(x + self.patch_size, W)

                patch = image[..., y1:y2, x1:x2]
                ph, pw = patch.shape[-2:]

                if ph < self.patch_size or pw < self.patch_size:
                    pad_h = self.patch_size - ph
                    pad_w = self.patch_size - pw
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                        mode="constant"
                    )

                patches.append(patch)
                coords.append((y1, y2, x1, x2, ph, pw))

        for i in range(0, len(patches), batch_size):
            batch = np.concatenate(patches[i:i+batch_size], axis=0)

            outputs = self.session.run_with_iobinding_numpy({self.input_name: batch})

            immasks = outputs[1]

            for j in range(immasks.shape[0]):
                y1, y2, x1, x2, ph, pw = coords[i + j]

                mask_output[..., y1:y2, x1:x2] += immasks[j:j+1, :, :ph, :pw]

                counter[..., y1:y2, x1:x2] += 1.0

        mask_output /= counter
        img_output  /= counter

        return mask_output

    def watermark_segment(self, image_path: str):
        image = self._preprocess(image_path)
        result = {}
        model_res = self._inference_adaptive(image)
        immask = model_res
        gen_immask = np.where(
            immask > 0.5,
            1.0,
            0.0
        ).astype(np.float32)

        result["mask"] = gen_immask
        return result
    

if __name__ == "__main__":
    import os
    from PIL import Image
    input_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "image.jpg")
    detection = SLBRSegment()
    detection.prepare(onnx_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "OnnxModels", "sr_segment.encmodel"))

    result = detection.watermark_segment(input_image)

    mask = result["mask"][0, 0]              # [H, W]
    mask = np.clip(mask, 0.0, 1.0)
    mask = (mask * 255.0).astype(np.uint8)

    Image.fromarray(mask).save("mask_output.png")
    