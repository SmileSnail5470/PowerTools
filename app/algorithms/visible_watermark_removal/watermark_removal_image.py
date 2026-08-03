import tempfile
import os
import cv2
import numpy as np
from app.algorithms.visible_watermark_removal.modules.sr_segment import SLBRSegment
from app.algorithms.visible_watermark_removal.modules.pt_segment import PatchWiperSegment
from app.algorithms.visible_watermark_removal.modules.pt_inpaint import PatchWiperInpaint
from app.algorithms.visible_watermark_removal.modules.cf_inpaint import CoordFillInpaint
from app.algorithms.visible_watermark_removal.modules.grig_inpaint import GRIGInpaint
from app.algorithms.visible_watermark_removal.modules.emdf_inpaint import EMDFInpaint
from app.algorithms.visible_watermark_removal.modules.lama_inpaint import LamaInpaint
from app.algorithms.visible_watermark_removal.modules.text_detection import detect_text_watermarks
from app.algorithms.visible_watermark_removal.modules.yolo_detecttion import YOLODetection
from app.algorithms.segment.inference import SegmentationInference
from app.algorithms.image_edit.general_edit.inference import ImageEditInference


class WatermarkSegment():
    def __init__(self, watermark_type, ai_detect_type, ai_interactive_type, ai_interactive_prompt, ai_interactive_boxes, watermark_confidence):
        self.watermark_type = watermark_type
        self.ai_detect_type = ai_detect_type
        self.ai_interactive_type = ai_interactive_type
        self.ai_interactive_prompt = ai_interactive_prompt
        self.ai_interactive_boxes = ai_interactive_boxes
        self.watermark_confidence = watermark_confidence

    def _split_mask_to_regions(self, mask: np.ndarray):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        regions = []
        for i in range(1, num_labels):
            region = (labels == i).astype(np.uint8) * 255
            regions.append(region)
        return regions

    def _region_is_rotated(self, region_mask, angle_thresh=10):
        ys, xs = np.where(region_mask > 0)
        if len(xs) < 20:
            return False
        
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        angle = abs(angle) % 360
        if angle > 180:
            angle = 360 - angle
        angle = min(
            abs(angle - 0),
            abs(angle - 90),
            abs(angle - 180),
        )
        return angle > angle_thresh

    def _merge_text_yolo_masks(self, text_mask, yolo_mask):
        final_mask = np.zeros_like(text_mask)
        regions = self._split_mask_to_regions(text_mask)

        for region in regions:
            if self._region_is_rotated(region):
                merged = region
            else:
                merged = cv2.bitwise_or(region, yolo_mask)

            final_mask = cv2.bitwise_or(final_mask, merged)

        return final_mask

    def segment(
            self, 
            image_path: str, 
            sr_onnx_path: str, 
            pt_onnx_path: str, 
            text_detection_onnx_path: str, 
            yolo_detection_onnx_path: str, 
            segment_onnx_dir: str,
            **kwargs
        ):
        if self.ai_detect_type == "ai_auto_detect":
            if self.watermark_type == "text":
                mask = detect_text_watermarks(
                    input_path=image_path,
                    limit_side_len = kwargs.get("limit_side_len", 960),
                    limit_type = kwargs.get("limit_type", "max"),
                    det_thresh = kwargs.get("det_thresh", 0.3),
                    det_box_thresh = kwargs.get("det_box_thresh", 0.6),
                    unclip_ratio = kwargs.get("unclip_ratio", 1.5),
                    score_mode = kwargs.get("score_mode", "fast"),
                    det_box_type = kwargs.get("det_box_type", "quad"),
                    onnx_path = text_detection_onnx_path
                )
                watermark_detector = YOLODetection()
                watermark_detector.prepare(
                    confidence=kwargs.get("confidence", 0.03), 
                    iou_threshold=kwargs.get("iou_threshold", 0.2), 
                    onnx_path=yolo_detection_onnx_path
                )
                _, output_masks, combined_info = watermark_detector.detect_watermarks(image_path=image_path)
                if combined_info["num_detections"] > 0:
                    yolo_mask = output_masks[0, 0]              # [H, W]
                    yolo_mask = np.clip(yolo_mask, 0.0, 1.0)
                    yolo_mask = (yolo_mask * 255.0).astype(np.uint8)
                    mask = self._merge_text_yolo_masks(mask, yolo_mask)
            elif self.watermark_type == "subtitle":
                mask = detect_text_watermarks(
                    input_path=image_path,
                    limit_side_len = kwargs.get("limit_side_len", 960),
                    limit_type = kwargs.get("limit_type", "max"),
                    det_thresh = kwargs.get("det_thresh", 0.3),
                    det_box_thresh = kwargs.get("det_box_thresh", 0.6),
                    unclip_ratio = kwargs.get("unclip_ratio", 1.5),
                    score_mode = kwargs.get("score_mode", "fast"),
                    det_box_type = kwargs.get("det_box_type", "quad"),
                    onnx_path = text_detection_onnx_path
                )
            else:
                slbr_segment = SLBRSegment()
                slbr_segment.prepare(onnx_path=sr_onnx_path)
                sl_mask = slbr_segment.watermark_segment(image_path=image_path)["mask"]  # [1, 1, H, W] 0~1 float
                sl_mask = (np.clip(sl_mask[0, 0], 0.0, 1.0) * 255).astype(np.uint8)      # [H, W]  0~255 int

                pt_segment = PatchWiperSegment()
                pt_segment.prepare(onnx_path=pt_onnx_path)
                pt_mask = pt_segment.watermark_segment(image_path=image_path)["mask"]    # [1, 1, H, W]  0~1 float
                pt_mask = (np.clip(pt_mask[0, 0], 0.0, 1.0) * 255).astype(np.uint8)      # [H, W]  0~255 int

                mask = np.clip((sl_mask + pt_mask), 0, 255).astype(np.uint8)
        elif self.ai_detect_type == "ai_interactive_detect":
            if self.ai_interactive_type == "semantic_detect":
                segment_instance = SegmentationInference(
                    model_dir=segment_onnx_dir,
                    prompt_mode="texts",
                    prompt_value=self.ai_interactive_prompt,
                    threshold=float(self.watermark_confidence)
                )
                segment_instance.prepare()
                mask = segment_instance.inference_image(input_image_path=image_path)
            else:
                prompt_value = []
                for item in self.ai_interactive_boxes:
                    prompt_value.append(",".join(str(i) for i in item))
                segment_instance = SegmentationInference(
                    model_dir=segment_onnx_dir,
                    prompt_mode="boxes",
                    prompt_value=prompt_value,
                    threshold=float(self.watermark_confidence)
                )
                segment_instance.prepare()
                mask = segment_instance.inference_image(input_image_path=image_path)

        return mask
    

class WatermarkInpaint():
    def __init__(
            self, 
            mask,  
            pt_inpaint_onnx_path, 
            cf_onnx_path, 
            lama_onnx_path,
            emdf_onnx_path,
            grig_onnx_path,
            general_edit_onnx_dir,
            model_type="lama",
            dilate_num=2,
        ):
        self.mask = mask  # [H, W] 0~255 int
        self.model_type = model_type.lower()
        self.dilate_num = dilate_num
        self.pt_inpaint_onnx_path = pt_inpaint_onnx_path
        self.cf_onnx_path = cf_onnx_path
        self.lama_onnx_path = lama_onnx_path
        self.emdf_onnx_path = emdf_onnx_path
        self.grig_onnx_path = grig_onnx_path
        self.general_edit_onnx_dir = general_edit_onnx_dir

    def _save_watermark_removed_image(self, image, output_path):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)

    def _process_image_with_general_edit(self, image_path):
        general_edit_inpaint = ImageEditInference(
            model_dir=self.general_edit_onnx_dir,
            dilate_num=self.dilate_num
        )
        result = general_edit_inpaint.infer(
            prompt="Remove watermark",
            input_path=image_path, 
            mask=self.mask
        )
        return result

    def _process_image_with_lama(self, image_path):
        lama_inpaint = LamaInpaint()
        lama_inpaint.prepare(onnx_path=self.lama_onnx_path)

        result = lama_inpaint.inpaint(image_path, mask=self.mask, dilate_num=self.dilate_num)  # [1, 3, H, W] 0~1 float
        result = (np.transpose(result[0], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)

        return result  # [H, W, 3] 0~255 float RGB
    
    def _process_image_with_patchwiper(self, image_path):
        pt_inpaint = PatchWiperInpaint()
        pt_inpaint.prepare(onnx_path=self.pt_inpaint_onnx_path)

        mask = self.mask.astype(np.float32) / 255.0
        mask = mask[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)

        result = pt_inpaint.inpaint(image_path, mask=mask, dilate_num=self.dilate_num)  # [1, 3, H, W] 0~1 float
        result = (np.transpose(result[0], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

        return result  # [H, W, 3] 0~255 float RGB
    
    def _process_image_with_coordfill(self, image_path):
        cf_inpaint = CoordFillInpaint()
        cf_inpaint.prepare(onnx_path=self.cf_onnx_path)

        result = cf_inpaint.inpaint(image_path, mask=self.mask, dilate_num=self.dilate_num)

        return result  # [H, W, 3] 0~255 float RGB
    
    def _process_image_with_grig(self, image_path):
        grig_inpaint = GRIGInpaint()
        grig_inpaint.prepare(onnx_path=self.grig_onnx_path)
        result = grig_inpaint.inpaint(image_path, mask=self.mask, dilate_num=self.dilate_num)

        return result  # [H, W, 3] 0~255 float RGB

    def _process_image_with_emdf(self, image_path):
        emdf_inpaint = EMDFInpaint()
        emdf_inpaint.prepare(onnx_path=self.emdf_onnx_path)
        result = emdf_inpaint.inpaint(image_path, mask=self.mask, dilate_num=self.dilate_num)

        return result  # [H, W, 3] 0~255 float RGB

    def _process_image_with_transparent(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        transparent_image = np.zeros((h, w, 4), dtype=np.uint8)

        transparent_image[:, :, :3] = image

        # 默认全不透明
        transparent_image[:, :, 3] = 255

        transparent_image[self.mask > 0] = (0, 0, 0, 0)

        return transparent_image
    
    def _process_image_with_cv2(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.mask[..., np.newaxis]

        assert len(mask.shape) == 3 and mask.shape[2] == 1

        cur_res = cv2.inpaint(
            image[:, :, ::-1],
            mask,
            inpaintRadius=4,
            flags=cv2.INPAINT_TELEA,
        )
        return cv2.cvtColor(cur_res, cv2.COLOR_BGR2RGB)
    
    def inpaint(self, image_path: str, output_path: str):
        if self.model_type == "lama":
            img = self._process_image_with_lama(image_path=image_path)
        elif self.model_type == "patchwiper":
            img = self._process_image_with_patchwiper(image_path=image_path)
        elif self.model_type == "coordfill":
            img = self._process_image_with_coordfill(image_path=image_path)
        elif self.model_type == "cv2":
            img = self._process_image_with_cv2(image_path=image_path)
        elif self.model_type == "transparent":
            img = self._process_image_with_transparent(image_path=image_path)
        elif self.model_type == "grig":
            img = self._process_image_with_grig(image_path=image_path)
        elif self.model_type == "emdf":
            img = self._process_image_with_emdf(image_path=image_path)
        elif self.model_type == "general_edit":
            img = self._process_image_with_general_edit(image_path=image_path)
        else:
            raise Exception(f"not support {self.model_type}")
        
        self._save_watermark_removed_image(img, output_path=output_path)


class ImageWatermarkRemove():
    def __init__(self):
        pass

    def _save_mask_visualization(
            self,
            img_path,
            mask: np.ndarray,
            file_path: str,
        ):
        img = cv2.imread(img_path)
        overlay = img.copy()
        color = (255, 0, 0)
        b, g, r_ = color
        mask = mask > 0
        overlay[mask] = (
            overlay[mask] * 0.55
            + np.array([b, g, r_]) * 0.45
        ).astype(np.uint8)
        cv2.imwrite(file_path, overlay)

    def _read_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        # mask 是 (H, W) 值是 0/255
        mask = ((mask > 0) * 255).astype(np.uint8)
        return mask

    def run(
            self, 
            image_path, 
            output_path,
            sr_segment_onnx_path,
            pt_segment_onnx_path,
            pt_inpaint_onnx_path, 
            cf_onnx_path, 
            lama_onnx_path,
            emdf_onnx_path,
            grig_onnx_path,
            text_detection_onnx_path,
            yolo_detection_onnx_path,
            segment_onnx_dir,
            general_edit_onnx_dir,
            mask_path: str = "",
            refine_type: str = "patchwiper",                    # patchwiper/lama/transparent/cv2/coordfill/grig/emdf
            watermark_type: str = "all",                        # text / all
            ai_detect_type: str = "ai_interactive_detect",      # ai_interactive_detect/ai_auto_detect
            ai_interactive_type: str = "semantic_detect",       # semantic_detect/space_detect
            ai_interactive_prompt: str = "watermark",
            ai_interactive_boxes: list = [],
            watermark_confidence: float = 0.5,
            watermark_boxes: list = [],
            dilate_num: int = 2,
            **kwargs
        ):
        progress_cb = kwargs.pop("progress_cb", None)
        if progress_cb is not None:
            progress_cb("MaskStart", "")
        if not mask_path:
            mask = WatermarkSegment(watermark_type, ai_detect_type, ai_interactive_type, ai_interactive_prompt, ai_interactive_boxes, watermark_confidence).segment(
                image_path=image_path,
                sr_onnx_path=sr_segment_onnx_path,
                pt_onnx_path=pt_segment_onnx_path,
                text_detection_onnx_path=text_detection_onnx_path,
                yolo_detection_onnx_path=yolo_detection_onnx_path,
                segment_onnx_dir=segment_onnx_dir,
                **kwargs
            )
            if watermark_boxes and not (ai_detect_type == "ai_interactive_detect" and ai_interactive_type == "space_detect"):
                h, w = mask.shape
                mask_image = np.zeros((h, w), dtype=np.uint8)
                for bbox in watermark_boxes:
                    xmin, ymin, xmax, ymax = bbox
                    mask_image[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
                mask = mask_image
        else:
            mask = self._read_mask(mask_path=mask_path)
        if progress_cb is not None:
            if "visible_mask_path" in kwargs and kwargs["visible_mask_path"]:
                mask_tmp_path = kwargs["visible_mask_path"]
            else:
                mask_tmp_path = "{0}_mask.png".format(output_path.rsplit(".", 1)[0])
            self._save_mask_visualization(
                img_path=image_path,
                mask=mask,
                file_path=mask_tmp_path,
            )
            progress_cb("MaskCompleted", mask_tmp_path)

        image_inpainting = WatermarkInpaint(
            mask=mask,
            pt_inpaint_onnx_path=pt_inpaint_onnx_path,
            cf_onnx_path=cf_onnx_path,
            lama_onnx_path=lama_onnx_path,
            emdf_onnx_path=emdf_onnx_path,
            grig_onnx_path=grig_onnx_path,
            general_edit_onnx_dir=general_edit_onnx_dir,
            model_type=refine_type,
            dilate_num=dilate_num,
        )
        image_inpainting.inpaint(image_path=image_path, output_path=output_path)
        if progress_cb is not None:
            progress_cb("WaterRemoved", "")