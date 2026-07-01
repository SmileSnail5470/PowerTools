import os
import numpy as np
from PIL import Image
from app.algorithms import general_inference_session, general_provider, general_session, ORTEnvironment
ORTEnvironment.initialize()
from app.algorithms.blind_watermark_addition.ecc_utils import HammingECC


class ImageBlindWatermarkEmbed():
    def __init__(self):
        pass

    def _create_predictor(self):
        session_options = general_session()
        providers, provider_options = general_provider(use_cpu=True)
        self.session = general_inference_session(
            self.onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )
        inputs = self.session.get_inputs()
        self.image_input = inputs[0].name
        self.msg_input  = inputs[1].name

        self.output_name = self.session.get_outputs()[0].name

    def prepare(self, onnx_path: str = None):
        self.onnx_path = onnx_path
        self._create_predictor()

    def watermark_addition(self, input_image_path: str, output_file: str, message: str):
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # load image
        imgs = Image.open(input_image_path, "r").convert("RGB")
        imgs = np.array(imgs)
        imgs = np.transpose(imgs, (2, 0, 1)).astype(np.float32) / 255.0
        imgs = imgs[np.newaxis, ...] # [1, C, H, W] 0~1.0

        # Watermark embedding
        ecc = HammingECC()
        if os.getenv("BLIND_WATERMARK_CHARSET") is not None:
            charset = os.getenv("BLIND_WATERMARK_CHARSET")
            ecc.CHARSET = charset
        msgs, _ = ecc.str_to_tensor(message)

        imgs_ws = self.session.run(
            [self.output_name],
            {
                self.image_input: imgs,
                self.msg_input: msgs
            }
        )
        imgs_w = imgs_ws[0]

        # save
        img = imgs_w[0]
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img_pil = Image.fromarray(img)
        img_pil.save(output_file)


class ImageBlindWatermarkDetect():
    def __init__(self):
        pass

    def _create_predictor(self):
        session_options = general_session()
        providers, provider_options = general_provider(use_cpu=True)
        self.session = general_inference_session(
            self.onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )
        inputs = self.session.get_inputs()
        self.image_input = inputs[0].name

        self.output_name = self.session.get_outputs()[0].name

    def prepare(self, onnx_path: str = None):
        self.onnx_path = onnx_path
        self._create_predictor()

    def watermark_extraction(self, input_image_path):
        # load image
        imgs = Image.open(input_image_path, "r").convert("RGB")
        imgs = np.array(imgs)
        imgs = np.transpose(imgs, (2, 0, 1)).astype(np.float32) / 255.0
        imgs = imgs[np.newaxis, ...] # [1, C, H, W] 0~1.0

        # Watermark detection
        preds = self.session.run(
            [self.output_name],
            {
                self.image_input: imgs
            }
        )[0]

        ecc = HammingECC()
        if os.getenv("BLIND_WATERMARK_CHARSET") is not None:
            charset = os.getenv("BLIND_WATERMARK_CHARSET")
            ecc.CHARSET = charset
        preds_str, preds = ecc.tensor_to_string(preds)
        metrics = {
            "file": input_image_path,
            "preds": preds_str
        }
        return metrics
    

if __name__ == "__main__":
    embed_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnxmodel", "pixelseal_image_embed.encmodel")
    detect_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnxmodel", "pixelseal_image_detect.encmodel")
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "imgs", "test.jpg")
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "test.jpg")

    embed_instance = ImageBlindWatermarkEmbed()
    embed_instance.prepare(onnx_path=embed_onnx_path)
    embed_instance.watermark_addition(input_image_path=image_path, output_file=output_path, message="HELLO,1234")
    print(f"embed {output_path} success.")

    detect_instance = ImageBlindWatermarkDetect()
    detect_instance.prepare(onnx_path=detect_onnx_path)
    res = detect_instance.watermark_extraction(input_image_path=output_path)
    print(res["preds"])