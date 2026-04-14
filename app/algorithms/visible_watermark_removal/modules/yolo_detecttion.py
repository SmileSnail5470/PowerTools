import platform
import sys

from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort
ort.preload_dlls(directory="")


def prepare_input(img, size):
    # original size
    orig_w, orig_h = img.size

    # calc scale ratio
    ratio = min(size / orig_w, size / orig_h)
    
    # calc new size
    scaled_w, scaled_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))

    # scale
    scaled_img = img.resize((scaled_w, scaled_h), resample=Image.Resampling.BILINEAR)

    # Calculate padding
    dh = 0 if (scaled_h % 32) == 0 else 32 - (scaled_h % 32)
    dw = 0 if (scaled_w % 32) == 0 else 32 - (scaled_w % 32)

    # Pad
    inp = np.full(
        (scaled_h + dh, scaled_w + dw, 3), 114, dtype=np.float32
    )
    inp[:scaled_h, :scaled_w, :] = np.array(scaled_img)

    # Scale input pixel values to 0 to 1
    inp = inp / 255.0
    inp = inp.transpose(2, 0, 1)
    inp = inp[None, :, :, :]

    return inp, (orig_w, orig_h), (scaled_w, scaled_h)

def post_process(outp, conf_thres=0.7, iou_thres=0.5):
    preds = np.squeeze(outp[0]).T

    # Remove low-conf preds
    scores = np.max(preds[:, 4:], axis=1)
    keep = scores > conf_thres

    # get boxes, scores and class_ids
    preds = preds[keep, :]
    boxes = preds[:, :4]
    boxes = xywh2xyxy(boxes)

    scores = np.max(preds[:, 4:], axis=1)
    
    class_ids = np.argmax(preds[:, 4:], axis=1)

    # do multiclass nms
    indices = multiclass_nms(boxes, scores, class_ids, iou_thres=iou_thres)
    
    return boxes[indices], scores[indices], class_ids[indices]

def post_process_pose(outp, conf_thres=0.7, iou_thres=0.5):
    preds = np.squeeze(outp[0]).T

    # Remove low-conf preds
    scores = np.max(preds[:, 4:5], axis=1)
    keep = scores > conf_thres

    # get boxes, scores, class_ids and kps
    preds = preds[keep, :]
    boxes = preds[:, :4]
    boxes = xywh2xyxy(boxes)

    scores = np.max(preds[:, 4:5], axis=1)
    class_ids = np.argmax(preds[:, 4:5], axis=1)

    kps = preds[:, 5:]
    kps = kps.reshape((-1, 17, 3))

    # do multiclass nms
    indices = multiclass_nms(boxes, scores, class_ids, iou_thres=iou_thres)
    
    return boxes[indices], scores[indices], class_ids[indices], kps[indices]

def xywh2xyxy(boxes):
    new_boxes = np.copy(boxes)
    new_boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    new_boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    new_boxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    new_boxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    
    return new_boxes

def multiclass_nms(boxes, scores, class_ids, iou_thres=0.5):
    unique_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_thres=iou_thres)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def nms(boxes, scores, iou_thres=0.5):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep 

def scale_boxes(boxes, orig_size, scaled_size):
    ox, oy, sx, sy = *orig_size, *scaled_size
    scale = np.array([ox/sx, oy/sy, ox/sx, oy/sy])
    boxes = boxes * scale
    return boxes

def scale_kps(kps, orig_size, scaled_size):
    ox, oy, sx, sy = *orig_size, *scaled_size
    scale = np.array([ox/sx, oy/sy])
    kps[:,:,:2] = kps[:,:,:2] * scale
    return kps

def parse_detections(boxes, scores, class_ids):
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append({
            'bbox': [int(b) for b in box],
            'score': float(round(score, 3)),
            'class_id': int(class_id)
        })
    return detections

def parse_detections_w_kps(boxes, scores, class_ids, kps):
    detections = []
    for box, score, class_id, kp in zip(boxes, scores, class_ids, kps):
        detections.append({
            'bbox': [int(b) for b in box],
            'score': float(round(score, 3)),
            'class_id': int(class_id),
            'kps': [k.tolist() for k in kp]
        })
    return detections


class YOLODetection():
    def __init__(self):
        pass

    def _create_predictor(self, onnx_path):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        sess = ort.InferenceSession(
            onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )
        inputs = sess.get_inputs()
        return (
            sess,
            inputs[0] if len(inputs) == 1 else [vo.name for vo in inputs]
        )

    def prepare(self, confidence=0.05, iou_threshold=0.2, onnx_path: str = None):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.onnx_path = onnx_path

    def _create_detection_image(self, original_image, detections):
        if detections is None:
            return original_image

        img = np.array(original_image)

        for box in detections:
            x1, y1, x2, y2 = box["bbox"]
            conf = box["score"]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label with confidence
            label = f"Watermark {conf:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Draw label background
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return Image.fromarray(img)

    def _create_watermark_mask(self, image_size, detections, mask_path: str = ""):
        width, height = image_size
        det_mask = np.zeros((height, width), dtype=np.float32)

        if detections is not None:
            for box in detections:
                x1, y1, x2, y2 = box["bbox"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    det_mask[y1:y2, x1:x2] = 1.0

        if not mask_path:
            return det_mask

        user_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if user_mask.ndim == 3 and user_mask.shape[2] == 1:
            user_mask = user_mask[:, :, 0]
        assert user_mask.shape == (height, width), \
            f"Mask shape {user_mask.shape} does not match image size {(height, width)}"

        user_mask = user_mask.astype(np.float32) / 255.0
        user_mask = (user_mask > 0.5).astype(np.float32)

        final_mask = det_mask * user_mask

        return final_mask

    def _format_detection_info(self, detections):
        watermark_info = {}
        watermark_info["num_detections"] = 0
        watermark_info["average_confidence"] = 0.0
        watermark_info["boxes"] = []

        if detections is None:
            return watermark_info

        num_detections = len(detections)
        confidences = [box["score"] for box in detections]
        avg_confidence = np.mean(confidences) if confidences else 0

        for i, (box, conf) in enumerate(zip(detections, confidences)):
            x1, y1, x2, y2 = box["bbox"]
            watermark_info["boxes"].append({"confidence": conf, "coordinates": [x1, y1, x2, y2]})

        watermark_info["num_detections"] = num_detections
        watermark_info["average_confidence"] = avg_confidence

        return watermark_info

    def detect_watermarks(self, image_path, mask_path: str = ""):
        pil_image = Image.open(image_path).convert("RGB")
        if max(pil_image.size) <= 640:
            resolution = 640
        elif max(pil_image.size) <= 1280:
            resolution = 1280
        else:
            resolution = 1920

        sess, _ = self._create_predictor(self.onnx_path)

        inp, orig_size, scaled_size = prepare_input(pil_image, resolution)

        outp = sess.run(['output0'], {'images': inp})

        boxes, scores, class_ids = post_process(outp, conf_thres=self.confidence, iou_thres=self.iou_threshold)

        boxes = scale_boxes(boxes, orig_size, scaled_size)

        detections = parse_detections(boxes, scores, class_ids)

        detection_image = self._create_detection_image(pil_image, detections)
        watermark_mask = self._create_watermark_mask(pil_image.size, detections, mask_path)
        combined_info = self._format_detection_info(detections)

        # Convert back to tensors (N, C, H, W)
        detection_array = np.array(detection_image).astype(np.float32)
        output_images = detection_array / 255.0
        output_images = np.transpose(output_images, (2, 0, 1))
        output_images = np.expand_dims(output_images, axis=0)

        output_masks = np.asarray(watermark_mask, dtype=np.float32)
        output_masks = np.expand_dims(output_masks, axis=(0, 1))

        return (output_images, output_masks, combined_info)
    

if __name__ == "__main__":
    import os
    input_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "image.jpg")
    detection = YOLODetection()
    detection.prepare(onnx_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "OnnxModels", "yolo.onnx"))
    
    output_images, output_masks, combined_info = detection.detect_watermarks(
        image_path=input_image,
        mask_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "mask.png"),
    )
    img = output_images[0]                 # [3, H, W]
    img = np.transpose(img, (1, 2, 0))     # CHW → HWC
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)

    Image.fromarray(img).save("detection_output.png")
    mask = output_masks[0, 0]              # [H, W]
    mask = np.clip(mask, 0.0, 1.0)
    mask = (mask * 255.0).astype(np.uint8)

    Image.fromarray(mask).save("mask_output.png")

    print(combined_info)
        