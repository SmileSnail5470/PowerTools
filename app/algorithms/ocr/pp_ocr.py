import copy
import math
import os
import random
import cv2
import numpy as np
from PIL import ImageDraw, Image, ImageFont
from app.algorithms.ocr.modules.text_detection import TextDetector
from app.algorithms.ocr.modules.text_recognition import TextRecognizer
from app.algorithms.ocr.modules.textline_classifier import TextClassifier


def draw_ocr_box_txt(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "pp_rec", "simfang.ttf"),
    ):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        img_right_text = _draw_box_txt_fine((w, h), box, txt, font_path)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


def _draw_box_txt_fine(img_size, box, txt, font_path):
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = _create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = _create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text


def _create_font(txt, sz, font_path):
    font_size = int(sz[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getlength(txt)
    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


class OCR():
    def __init__(self):
        self.text_detector = TextDetector()
        self.text_recognizer = TextRecognizer()
        self.text_classifier = TextClassifier()

    def prepare(
            self,
            limit_side_len = 960,
            limit_type = "max",
            det_thresh = 0.3,
            det_box_thresh = 0.6,
            unclip_ratio = 1.5,
            score_mode = "fast",
            det_box_type = "quad",
            drop_score = 0.5,
            det_onnx_path = None,
            rec_onnx_path = None,
            cls_onnx_path = None,
            progress_cb = None
        ):
        self.det_box_type = det_box_type
        self.drop_score = drop_score
        self.text_detector.prepare(
            limit_side_len = limit_side_len,
            limit_type = limit_type,
            det_thresh = det_thresh,
            det_box_thresh = det_box_thresh,
            unclip_ratio = unclip_ratio,
            score_mode = score_mode,
            det_box_type = det_box_type,
            onnx_path = det_onnx_path
        )
        self.text_recognizer.prepare(onnx_path=rec_onnx_path)
        self.text_classifier.prepare(onnx_path=cls_onnx_path)
        self.progress_cb = progress_cb

    def predict(self, image_path, output_path=None, use_cls=True, gen_visualize=False):
        if self.progress_cb is not None:
            self.progress_cb("OCRStart", "")
        img = cv2.imread(image_path)
        ori_im = img.copy()
        dt_boxes, _ = self.text_detector(img)

        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        if use_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)

        rec_res, _ = self.text_recognizer(img_crop_list)
        
        if gen_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]
            draw_img = draw_ocr_box_txt(
                image,
                dt_boxes,
                txts,
                scores,
                drop_score=self.drop_score,
            )
            cv2.imwrite(output_path, draw_img[:, :, ::-1])

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        if self.progress_cb is not None:
            self.progress_cb("OCRCompleted", "")
        return filter_boxes, filter_rec_res
    

if __name__ == "__main__":
    import os
    image_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assess", "test1.jpg")
    output_file = "./out.png"
    det_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources", "deps", "ocr", "pp_ocr_det.encmodel")
    rec_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources", "deps", "ocr", "pp_ocr_rec.encmodel")
    cls_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources", "deps", "ocr", "pp_lcnet_x1_0_textline_ori.encmodel")

    ocr = OCR()
    ocr.prepare(det_onnx_path=det_onnx_path, rec_onnx_path=rec_onnx_path, cls_onnx_path=cls_onnx_path)
    filter_boxes, filter_rec_res = ocr.predict(image_path=image_file, output_path=output_file, gen_visualize=True)
    print(filter_boxes, filter_rec_res)