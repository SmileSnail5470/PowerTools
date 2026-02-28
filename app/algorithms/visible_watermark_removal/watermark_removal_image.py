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


class WatermarkSegment():
    def __init__(self, watermark_type = "all"):
        self.watermark_type = watermark_type

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
            **kwargs
        ):
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

    def _save_watermark_removed_image(self, image, output_path):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)

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
        else:
            raise Exception(f"not support {self.model_type}")
        
        self._save_watermark_removed_image(img, output_path=output_path)


class ImageWatermarkRemove():
    def __init__(self):
        pass

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
            mask_path: str = "",
            refine_type: str = "patchwiper",  # patchwiper/lama/transparent/cv2/coordfill/grig/emdf
            watermark_type: str = "all",      # text / all
            dilate_num: int = 2,
            **kwargs
        ):
        if not mask_path:
            # ai 选择水印掩码
            mask = WatermarkSegment(watermark_type=watermark_type).segment(
                image_path=image_path,
                sr_onnx_path=sr_segment_onnx_path,
                pt_onnx_path=pt_segment_onnx_path,
                text_detection_onnx_path=text_detection_onnx_path,
                yolo_detection_onnx_path=yolo_detection_onnx_path,
                **kwargs
            )
        else:
            # 读取人工标注的水印掩码
            mask = self._read_mask(mask_path=mask_path)

        image_inpainting = WatermarkInpaint(
            mask=mask,
            pt_inpaint_onnx_path=pt_inpaint_onnx_path,
            cf_onnx_path=cf_onnx_path,
            lama_onnx_path=lama_onnx_path,
            emdf_onnx_path=emdf_onnx_path,
            grig_onnx_path=grig_onnx_path,
            model_type=refine_type,
            dilate_num=dilate_num,
        )
        image_inpainting.inpaint(image_path=image_path, output_path=output_path)

        if refine_type not in ["patchwiper"]:
            return
        
        remain_watermark = WatermarkSegment(watermark_type=watermark_type).segment(
            image_path=output_path,
            sr_onnx_path=sr_segment_onnx_path,
            pt_onnx_path=pt_segment_onnx_path,
            text_detection_onnx_path=text_detection_onnx_path,
            yolo_detection_onnx_path=yolo_detection_onnx_path,
            **kwargs
        )
        
        remain_watermark = ((remain_watermark.astype(np.float32) / 255.0) > 0.5).astype(np.float32) * \
                            ((mask.astype(np.float32) / 255.0) > 0.5).astype(np.float32)

        new_mask = remain_watermark
        
        watermark_detector = YOLODetection()
        watermark_detector.prepare(
            confidence=kwargs.get("confidence", 0.03), 
            iou_threshold=kwargs.get("iou_threshold", 0.2), 
            onnx_path=yolo_detection_onnx_path
        )
        _, output_masks, combined_info = watermark_detector.detect_watermarks(
            image_path=output_path,
            mask_path=mask_path
        )
        if combined_info["num_detections"] > 0:
            new_mask = output_masks[0, 0]              # [H, W]
            new_mask = np.clip(new_mask, 0.0, 1.0)
            new_mask = (new_mask * 255.0).astype(np.uint8)
            new_mask = ((new_mask.astype(np.float32) / 255.0) > 0.5).astype(np.float32) * \
                        ((mask.astype(np.float32) / 255.0) > 0.5).astype(np.float32)
            new_mask = np.maximum(((new_mask.astype(np.float32) / 255.0) > 0.5).astype(np.float32), remain_watermark)
            new_mask = (new_mask * 255).astype(np.uint8)

        if not np.any(new_mask):
            return
        
        image_inpainting = WatermarkInpaint(
            mask=new_mask,
            pt_inpaint_onnx_path=pt_inpaint_onnx_path,
            cf_onnx_path=cf_onnx_path,
            lama_onnx_path=lama_onnx_path,
            emdf_onnx_path=emdf_onnx_path,
            grig_onnx_path=grig_onnx_path,
            model_type="emdf",
            dilate_num=dilate_num,
        )
        image_inpainting.inpaint(image_path=output_path, output_path=output_path)


if __name__ == "__main__":
    import os
    import time
    start_time = time.time()
    refine_type = "patchwiper"
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assess", "256_test.png")
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"output_{refine_type}.png")
    water_remove = ImageWatermarkRemove()
    water_remove.run(
        image_path=input_path, 
        output_path=out_path,
        sr_segment_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "sr_segment.onnx"),
        pt_segment_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "pt_segment.onnx"),
        pt_inpaint_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "pt_inpaint.onnx"), 
        cf_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "cf.onnx"), 
        lama_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "lama_fp32.onnx"),
        emdf_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "emdf_inpaint.onnx"),
        grig_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "grig_inpaint.onnx"),
        text_detection_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "pp_ocr_det.onnx"),
        yolo_detection_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OnnxModels", "yolo.onnx"),
        mask_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assess", "256_test_mask.png"),
        refine_type=refine_type,
        watermark_type="text",
        dilate_num=2
    )
    print(f"Cost {time.time() - start_time}")