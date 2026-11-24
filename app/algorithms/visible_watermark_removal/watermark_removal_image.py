import os
import tempfile
import tempfile
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
from app.algorithms.private.object_segmentation.inference import create_inference as create_segmentation_inference
from app.algorithms.private.image_edit.inference import create_inference as create_edit_inference
from app.algorithms.private.image_restoration.inference import create_inference as create_restoration_inference
from app.algorithms.private.object_remove.inference import create_inference as create_object_remove_inference
from app.algorithms.visible_watermark_removal.params_config import ImageWatermarkRemoveConfig, RestorationRemoveConfig, DetectionConfig, SegmentModelPaths


class WatermarkSegment():
    def __init__(self, watermark_detect_params: DetectionConfig):
        self.watermark_ai_detect_method = watermark_detect_params.watermark_ai_detect_method
        self.watermark_ai_detect_prompt = watermark_detect_params.watermark_ai_detect_prompt
        self.watermark_ai_detect_pos_boxes = watermark_detect_params.watermark_ai_detect_pos_boxes
        self.watermark_ai_detect_neg_boxes = watermark_detect_params.watermark_ai_detect_neg_boxes
        self.watermark_ai_detect_confidence = watermark_detect_params.watermark_ai_detect_confidence
        self.watermark_ai_text_detection_result_mode = watermark_detect_params.watermark_ai_text_detection_result_mode

    def segment(
            self, 
            image_path: str, 
            sr_onnx_path: str, 
            pt_onnx_path: str, 
            text_detection_onnx_path: str, 
            segment_onnx_dir: str,
            **kwargs
        ):
        if self.watermark_ai_detect_method == "text_watermark":
            if self.watermark_ai_text_detection_result_mode == "detection":
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
                text_mask, text_boxes = detect_text_watermarks(
                    input_path=image_path,
                    limit_side_len = kwargs.get("limit_side_len", 960),
                    limit_type = kwargs.get("limit_type", "max"),
                    det_thresh = kwargs.get("det_thresh", 0.3),
                    det_box_thresh = kwargs.get("det_box_thresh", 0.6),
                    unclip_ratio = kwargs.get("unclip_ratio", 1.5),
                    score_mode = kwargs.get("score_mode", "fast"),
                    det_box_type = kwargs.get("det_box_type", "quad"),
                    onnx_path = text_detection_onnx_path,
                    return_boxes=True,
                )
                if text_boxes:
                    segment_instance = create_segmentation_inference("image", model_dir=segment_onnx_dir, threshold=0.3)
                    mask = segment_instance.infer(
                        input_path=image_path,
                        text=None,
                        threshold=0.3,
                        box=text_boxes,
                        negative_box=None
                    )
                else:
                    mask = text_mask
        elif self.watermark_ai_detect_method == "general_watermark":
            slbr_segment = SLBRSegment()
            slbr_segment.prepare(onnx_path=sr_onnx_path)
            sl_mask = slbr_segment.watermark_segment(image_path=image_path)["mask"]  # [1, 1, H, W] 0~1 float
            sl_mask = (np.clip(sl_mask[0, 0], 0.0, 1.0) * 255).astype(np.uint8)      # [H, W]  0~255 int

            pt_segment = PatchWiperSegment()
            pt_segment.prepare(onnx_path=pt_onnx_path)
            pt_mask = pt_segment.watermark_segment(image_path=image_path)["mask"]    # [1, 1, H, W]  0~1 float
            pt_mask = (np.clip(pt_mask[0, 0], 0.0, 1.0) * 255).astype(np.uint8)      # [H, W]  0~255 int

            mask = np.clip((sl_mask + pt_mask), 0, 255).astype(np.uint8)
        elif self.watermark_ai_detect_method == "semantic_detect":
            segment_instance = create_segmentation_inference(
                "image",
                model_dir=segment_onnx_dir,
                threshold=float(self.watermark_ai_detect_confidence)
            )
            mask = segment_instance.infer(
                input_path=image_path,
                text=self.watermark_ai_detect_prompt,
                threshold=self.watermark_ai_detect_confidence,
                box=self.watermark_ai_detect_pos_boxes,
                negative_box=self.watermark_ai_detect_neg_boxes
            )
        elif self.watermark_ai_detect_method == "space_detect":
            segment_instance = create_segmentation_inference(
                "image",
                model_dir=segment_onnx_dir,
                threshold=float(self.watermark_ai_detect_confidence)
            )
            mask = segment_instance.infer(
                input_path=image_path,
                text=None,
                threshold=self.watermark_ai_detect_confidence,
                box=self.watermark_ai_detect_pos_boxes,
                negative_box=self.watermark_ai_detect_neg_boxes
            )
        else:
            raise Exception(f"Not support watermark_ai_detect_method {self.watermark_ai_detect_method}")

        return mask
    

class WatermarkInpaint():
    def __init__(
            self, 
            mask=None,  
            pt_inpaint_onnx_path="", 
            cf_onnx_path="", 
            lama_onnx_path="",
            emdf_onnx_path="",
            grig_onnx_path="",
            general_edit_onnx_dir="",
            image_restoration_onnx_dir="",
            object_remove_onnx_dir="",
            restoration_prompt="",
            restoration_watermark_type="general",
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
        self.image_restoration_onnx_dir = image_restoration_onnx_dir
        self.object_remove_onnx_dir = object_remove_onnx_dir
        self.restoration_prompt = restoration_prompt
        self.restoration_watermark_type = restoration_watermark_type

    def _save_watermark_removed_image(self, image, output_path):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image)

    def _process_image_with_general_edit(self, image_path):
        general_edit_inpaint = create_edit_inference(
            "general_edit",
            model_dir=self.general_edit_onnx_dir,
            dilate_num=self.dilate_num
        )
        result = general_edit_inpaint.infer(
            input_path=image_path, 
            mask=self.mask,
            task_type="watermark_remove"
        )
        return result

    def _process_image_with_image_restoration(self, image_path):
        image_restoration_inpaint = create_restoration_inference(
            "image_restoration",
            model_dir=self.image_restoration_onnx_dir,
        )
        result = image_restoration_inpaint.infer(
            prompt=self.restoration_prompt,
            input_path=image_path,
            task_type="watermark_remove",
            watermark_type=self.restoration_watermark_type,
        )
        return result  # [H, W, 3] 0~255 uint8 RGB

    def _process_image_with_object_removal(self, image_path):
        object_remove_inpaint = create_object_remove_inference(
            "image",
            model_dir=self.object_remove_onnx_dir,
            dilate_num=self.dilate_num,
        )
        result = object_remove_inpaint.infer(
            input_path=image_path,
            mask=self.mask,
            task_type="object_remove",
        )
        return result  # [H, W, 3] 0~255 uint8 RGB

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
        elif self.model_type == "image_restoration":
            img = self._process_image_with_image_restoration(image_path=image_path)
        elif self.model_type == "object_removal":
            img = self._process_image_with_object_removal(image_path=image_path)
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

    @staticmethod
    def _get_tile_bounds(width: int, height: int, max_side: int = 1536):
        columns = max(1, (width + max_side - 1) // max_side)
        rows = max(1, (height + max_side - 1) // max_side)
        return [
            (column * width // columns, row * height // rows, (column + 1) * width // columns, (row + 1) * height // rows)
            for row in range(rows)
            for column in range(columns)
        ]

    def _detect_mask(self, image_path, segment_config: SegmentModelPaths, detection: DetectionConfig, kwargs):
        mask = WatermarkSegment(detection).segment(
            image_path=image_path,
            sr_onnx_path=segment_config.sr_segment_onnx_path,
            pt_onnx_path=segment_config.pt_segment_onnx_path,
            text_detection_onnx_path=segment_config.text_detection_onnx_path,
            segment_onnx_dir=segment_config.segment_onnx_dir,
            **detection.to_segment_kwargs(),
            **kwargs,
        )
        return mask

    def _detect_full_image_mask(self, image_path, segment_config: SegmentModelPaths, detection: DetectionConfig, kwargs):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        tile_bounds = self._get_tile_bounds(width, height)
        if len(tile_bounds) == 1 or not detection.enable_high_res_title_detect:
            mask = self._detect_mask(image_path, segment_config, detection, kwargs)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            with tempfile.TemporaryDirectory(prefix="watermark_detect_") as temp_dir:
                for index, (x1, y1, x2, y2) in enumerate(tile_bounds):
                    tile_image_path = os.path.join(temp_dir, f"tile_{index}.png")
                    if not cv2.imwrite(tile_image_path, image[y1:y2, x1:x2]):
                        raise OSError(f"Cannot write temporary image: {tile_image_path}")
                    tile_mask = self._detect_mask(tile_image_path, segment_config, detection, kwargs)
                    expected_shape = (y2 - y1, x2 - x1)
                    if tile_mask.shape != expected_shape:
                        raise ValueError(f"Detected mask dimensions {tile_mask.shape[::-1]} do not match tile dimensions {(x2 - x1, y2 - y1)}")
                    mask[y1:y2, x1:x2] = tile_mask
        if detection.watermark_ai_detect_pos_boxes:
            mask_image = np.zeros((height, width), dtype=np.uint8)
            for xmin, ymin, xmax, ymax in detection.watermark_ai_detect_pos_boxes:
                mask_image[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
            mask = mask_image
        return mask

    def _inpaint_tiles(self,
        output_path: str,
        image: np.ndarray,
        tile_bounds: list[tuple[int, int, int, int]],
        inpaint_kwargs: dict,
        mask: np.ndarray | None = None,
    ):
        height, width = image.shape[:2]
        if mask is not None and mask.shape != (height, width):
            raise ValueError(f"Mask dimensions {mask.shape[::-1]} do not match image dimensions {(width, height)}")
        result = image.copy()
        with tempfile.TemporaryDirectory(prefix="watermark_inpaint_") as temp_dir:
            for index, (x1, y1, x2, y2) in enumerate(tile_bounds):
                tile_mask = mask[y1:y2, x1:x2] if mask is not None else None
                if tile_mask is not None and not np.any(tile_mask):
                    continue
                tile_image_path = os.path.join(temp_dir, f"tile_{index}.png")
                tile_output_path = os.path.join(temp_dir, f"output_{index}.png")
                if not cv2.imwrite(tile_image_path, image[y1:y2, x1:x2]):
                    raise OSError(f"Cannot write temporary image: {tile_image_path}")
                tile_inpaint_kwargs = dict(inpaint_kwargs)
                if tile_mask is not None:
                    tile_inpaint_kwargs["mask"] = tile_mask
                WatermarkInpaint(**tile_inpaint_kwargs).inpaint(image_path=tile_image_path, output_path=tile_output_path)
                tile_result = cv2.imread(tile_output_path, cv2.IMREAD_UNCHANGED)
                expected_shape = (y2 - y1, x2 - x1)
                if tile_result is None or tile_result.shape[:2] != expected_shape:
                    raise ValueError(f"Processed tile dimensions do not match expected dimensions {expected_shape}")
                result[y1:y2, x1:x2] = tile_result[:, :, :3]
        if not cv2.imwrite(output_path, result):
            raise OSError(f"Cannot write output image: {output_path}")

    def run(self, image_path: str, output_path: str, config: ImageWatermarkRemoveConfig, **kwargs):
        seg = config.segment
        inp = config.inpaint
        det = config.detection
        msk = config.mask
        rt = config.runtime
        enable_high_res_title = config.enable_high_res_title

        progress_cb = rt.progress_cb
        if progress_cb is not None:
            progress_cb("MaskStart", "")

        if not msk.mask_path:
            mask = self._detect_full_image_mask(image_path, seg, det, kwargs)
        else:
            mask = self._read_mask(mask_path=msk.mask_path)

        if progress_cb is not None:
            if "POWERTOOLS_TMPE_DIR" in os.environ and os.environ["POWERTOOLS_TMPE_DIR"]:
                name_prefix = os.path.basename(output_path.rsplit(".", 1)[0])
                mask_tmp_path = os.path.join(os.environ["POWERTOOLS_TMPE_DIR"], "tmpdir", "{0}_mask.png".format(name_prefix))
            elif kwargs.get("visible_mask_path"):
                mask_tmp_path = kwargs["visible_mask_path"]
            else:
                mask_tmp_path = "{0}_mask.png".format(output_path.rsplit(".", 1)[0])
            if not os.path.exists(os.path.dirname(mask_tmp_path)):
                os.makedirs(os.path.dirname(mask_tmp_path), exist_ok=True)
            self._save_mask_visualization(
                img_path=image_path,
                mask=mask,
                file_path=mask_tmp_path,
            )
            progress_cb("MaskCompleted", mask_tmp_path)

        inpaint_kwargs = dict(
            pt_inpaint_onnx_path=inp.pt_inpaint_onnx_path,
            cf_onnx_path=inp.cf_onnx_path,
            lama_onnx_path=inp.lama_onnx_path,
            emdf_onnx_path=inp.emdf_onnx_path,
            grig_onnx_path=inp.grig_onnx_path,
            general_edit_onnx_dir=inp.general_edit_onnx_dir,
            object_remove_onnx_dir=inp.object_remove_onnx_dir,
            model_type=msk.refine_type,
            dilate_num=msk.dilate_num,
        )
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        tile_bounds = self._get_tile_bounds(width, height)
        if len(tile_bounds) > 1 and enable_high_res_title:
            self._inpaint_tiles(output_path, image, tile_bounds, inpaint_kwargs, mask)
        else:
            WatermarkInpaint(mask=mask, **inpaint_kwargs).inpaint(image_path=image_path, output_path=output_path)
        if progress_cb is not None:
            progress_cb("WaterRemoved", "")

    def run_without_mask(self, image_path: str, output_path: str, config: RestorationRemoveConfig):
        params = config.params
        progress_cb = config.runtime.progress_cb

        if progress_cb is not None:
            progress_cb("WaterRemoveStart", "")

        inpaint_kwargs = dict(
            image_restoration_onnx_dir=config.restoration.image_restoration_onnx_dir,
            restoration_prompt=params.restoration_prompt,
            restoration_watermark_type=params.restoration_watermark_type,
            model_type=params.refine_type,
        )
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        tile_bounds = self._get_tile_bounds(width, height)
        if len(tile_bounds) > 1 and config.enable_high_res_title:
            self._inpaint_tiles(output_path, image, tile_bounds, inpaint_kwargs)
        else:
            WatermarkInpaint(**inpaint_kwargs).inpaint(image_path=image_path, output_path=output_path)
        if progress_cb is not None:
            progress_cb("WaterRemoved", "")