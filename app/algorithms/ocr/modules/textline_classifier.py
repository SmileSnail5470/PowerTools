import copy
import os
import platform
import sys
import yaml
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
import cv2
import numpy as np
import math
import time
import onnxruntime as ort
ort.preload_dlls(directory="")
from app.algorithms import general_inference_session, general_provider, general_session, ORTEnvironment
ORTEnvironment.initialize()


class ClsPostProcess(object):
    """Convert between text-label and text-index"""

    def __init__(self, label_list=None, key=None, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list
        self.key = key

    def __call__(self, preds, label=None, *args, **kwargs):
        if self.key is not None:
            preds = preds[self.key]

        label_list = self.label_list
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        pred_idxs = preds.argmax(axis=1)
        decode_out = [(label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = [(label_list[idx], 1.0) for idx in label]
        return decode_out, label


def create_predictor(onnx_path):
    session_options = general_session()
    providers, provider_options = general_provider()
    sess = general_inference_session(
        onnx_path,
        providers=providers,
        provider_options=provider_options,
        sess_options=session_options,
    )
    inputs = sess.get_inputs()
    return (
        sess,
        inputs[0] if len(inputs) == 1 else [vo.name for vo in inputs],
        None,
        None,
    )


def build_post_process(config: dict, global_config=None):
    support_dict = [
        "ClsPostProcess",
    ]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        "post process only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class


class TextClassifier(object):
    def __init__(self):
        pass

    def prepare(self, onnx_path):
        self.onnx_path = onnx_path
        
        cls_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "pp_lcnet_textline_ori")
        config_path = os.path.join(cls_model_dir, "inference.yml")
        with open(config_path, "rb") as file:
            model_config = yaml.load(file, Loader=yaml.SafeLoader)
        model_name = model_config.get("Global", {}).get("model_name", "")
        if model_name and model_name not in [
            "PP-LCNet_x1_0_textline_ori",
        ]:
            raise ValueError(
                f"{model_name} is not supported. Please check if the model is supported by the PaddleOCR wheel."
            )

        self.cls_image_shape = [3, 80, 160]
        self.cls_batch_num = 6
        self.cls_thresh = 0.9
        postprocess_params = {
            "name": "ClsPostProcess",
            "label_list": ["0", "180"],
        }
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            _,
        ) = create_predictor(onnx_path=self.onnx_path)

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [["", 0.0]] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            starttime = time.time()
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors, input_dict)
            prob_out = outputs[0]
            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if "180" in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(img_list[indices[beg_img_no + rno]], 1)
        return img_list, cls_res, elapse