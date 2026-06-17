import copy
import os
import platform
import re
import sys
import yaml
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
import cv2
import numpy as np
import math
import time
import onnxruntime as ort
ort.preload_dlls(directory="")
import uuid
from app.algorithms import general_inference_session


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""
    def __init__(self, character_dict_path, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                self.character_str.append(line)
        if use_space_char:
            self.character_str.append(" ")
        dict_character = list(self.character_str)
        if "arabic" in character_dict_path:
            self.reverse = True
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, selection):
        """
        Group the decoded characters and record the corresponding decoded positions.

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types of grouping words:
                        - 'cn': continuous chinese characters (e.g., 你好啊)
                        - 'en&num': continuous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
                        The remaining characters in text are treated as separators between groups (e.g., space, '(', ')', etc.).
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection == True)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            # Use \w with UNICODE flag to match letters (including accented chars like ä, ö, ü, é, etc.) and digits
            # Exclude underscore since \w includes it but we want to treat it as splitter
            elif bool(re.search(r"[\w]", char, re.UNICODE)) and char != "_":
                c_state = "en&num"
            else:
                c_state = "splitter"

            # Handle apostrophes in French words like "n'êtes"
            if char == "'" and state == "en&num":
                c_state = "en&num"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ):  # grouping floating number
                c_state = "en&num"
            if (
                char == "-" and state == "en&num"
            ):  # grouping word with '-', such as 'state-of-the-art'
                c_state = "en&num"

            if state == None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""
    def __init__(self, character_dict_path, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, return_word_box=False, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
            return_word_box=return_word_box,
        )
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs["wh_ratio_list"][rec_idx]
                max_wh_ratio = kwargs["max_wh_ratio"]
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


def _hash_cuda_gpu():
    if platform.system() != "Windows":
        return True
    cuda_path = r"C:\Program Files\NVIDIA Corporation"
    if os.path.exists(cuda_path):
        return True
    return False


def create_predictor(onnx_path):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available = ort.get_available_providers()
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    elif "CUDAExecutionProvider" in available and _hash_cuda_gpu():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{}, {}]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
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
        "CTCLabelDecode",
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


class TextRecognizer():
    def __init__(self):
        self._temp_dict_file = None
    
    def __del__(self):
        self.cleanup()

    def prepare(self, onnx_path):
        self.onnx_path = onnx_path

        rec_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "pp_rec")
        config_path = os.path.join(rec_model_dir, "inference.yml")
        with open(config_path, "rb") as file:
            model_config = yaml.load(file, Loader=yaml.SafeLoader)
        model_name = model_config.get("Global", {}).get("model_name", "")
        if model_name and model_name not in [
            "PP-OCRv5_server_rec",
        ]:
            raise ValueError(f"{model_name} is not supported. Please check if the model is supported by the PaddleOCR wheel.")
        
        rec_char_list = model_config.get("PostProcess", {}).get("character_dict", [])
        # 使用唯一名字的临时文件避免多线程竞争
        unique_id = str(uuid.uuid4())[:8]
        rec_char_dict_path = os.path.join(rec_model_dir, f"ppocr_keys_{unique_id}.txt")
        
        with open(rec_char_dict_path, "w", encoding="utf-8") as f:
            f.writelines([char + "\n" for char in rec_char_list])
        
        # 保存临时文件路径用于后续清理
        self._temp_dict_file = rec_char_dict_path

        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 6
        self.rec_algorithm = "SVTR_LCNet"
        postprocess_params = {
            "name": "CTCLabelDecode",
            "character_dict_path": rec_char_dict_path,
            "use_space_char": True,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.postprocess_params = postprocess_params
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = create_predictor(onnx_path=self.onnx_path)
        self.return_word_box = False
    
    def cleanup(self):
        if self._temp_dict_file and os.path.exists(self._temp_dict_file):
            try:
                os.remove(self._temp_dict_file)
                self._temp_dict_file = None
            except Exception as e:
                print(f"Warning: Failed to remove temp dict file {self._temp_dict_file}: {e}")

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        w = self.input_tensor.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = outputs[0]
            rec_result = self.postprocess_op(
                preds,
                return_word_box=self.return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        self.cleanup()
        return rec_res, time.time() - st