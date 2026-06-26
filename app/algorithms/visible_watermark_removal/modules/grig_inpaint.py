import os
import platform
import sys
import cv2
import numpy as np
import onnxruntime as ort
ort.preload_dlls(directory="")
from PIL import Image
from app.algorithms import general_inference_session, general_session, general_provider, ORTEnvironment
ORTEnvironment.initialize()


class GRIGInpaint():
    def __init__(self):
        pass

    def _create_predictor(self):
        session_options = general_session()
        if os.getenv("WATERMARK_REMOVAL_MEMORY_OPTIMATION", False):
            session_options.enable_cpu_mem_arena = False
            session_options.enable_mem_pattern = False
        providers, provider_options = general_provider()
        self.session = general_inference_session(
            self.onnx_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )
        inputs = self.session.get_inputs()
        self.image_input_name = inputs[0].name
        self.mask_input_name  = inputs[1].name

        self.output_name = self.session.get_outputs()[0].name

    def prepare(self, onnx_path: str = None):
        self.onnx_path = onnx_path
        self._create_predictor()

    def _pad(self, image: Image.Image, size=256):
        w, h = image.size
        pad_w = (size - w % size) % size
        pad_h = (size - h % size) % size

        padded = Image.new(image.mode, (w + pad_w, h + pad_h))
        padded.paste(image, (0, 0))
        return padded
    
    def _load_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        return image  # [H, W, C] 0~255 int

    def _tile_has_watermark(self, mask_tile, bin_thresh=0.5, min_area_ratio=0.001, min_area_abs=300, min_sum=80, max_gap=8):
        if mask_tile.ndim == 4:
            mask_tile = mask_tile[0, 0]
        elif mask_tile.ndim == 3:
            mask_tile = mask_tile[0]

        h, w = mask_tile.shape
        tile_area = h * w

        binary = (mask_tile > bin_thresh).astype(np.uint8)
        if binary.sum() < min_sum:
            return False
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        min_area = max(int(tile_area * min_area_ratio), min_area_abs)

        comps = []
        for i in range(1, num_labels):
            x, y, w0, h0, area = stats[i]
            if area < 20:
                continue
            comps.append({
                "id": i,
                "bbox": (x, y, w0, h0),
                "area": area,
                "size": max(w0, h0),
            })

        if not comps:
            return False
        
        def bbox_gap(a, b):
            xa, ya, wa, ha = a["bbox"]
            xb, yb, wb, hb = b["bbox"]
            gap_x = max(0, xb - (xa + wa), xa - (xb + wb))
            gap_y = max(0, yb - (ya + ha), ya - (yb + hb))
            return max(gap_x, gap_y)

        visited = [False] * len(comps)
        for i in range(len(comps)):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            cluster_area = comps[i]["area"]

            while stack:
                cur = stack.pop()
                for j in range(len(comps)):
                    if visited[j]:
                        continue
                    gap = bbox_gap(comps[cur], comps[j])
                    thresh = max(max_gap, int(0.2 * min(comps[cur]["size"], comps[j]["size"])))
                    if gap <= thresh:
                        visited[j] = True
                        stack.append(j)
                        cluster_area += comps[j]["area"]
            if cluster_area >= min_area:
                return True
        return False
    
    def _crop_box(self, image, mask, box, migrate=64):
        """
        image: [H, W, C] RGB
        mask: [H, W]
        box: [left,top,right,bottom]
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + migrate * 2
        h = box_h + migrate * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        if max(crop_mask.shape[:2]) > 256 and migrate > 32:
            return self._crop_box(image, mask, box, migrate=migrate//2)
        return crop_img, crop_mask, [l, t, r, b]
    
    def boxes_from_mask(self, mask: np.ndarray):
        """
        mask: (H, W)  0~255
        """
        height, width = mask.shape[:2]
        _, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        x1, y1 = width, height
        x2, y2 = 0, 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            x1 = min(x1, x)
            y1 = min(y1, y)
            x2 = max(x2, x + cw)
            y2 = max(y2, y + ch)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        return [np.array([x1, y1, x2, y2], dtype=int)]
    
    def split_mask_to_regions(self, mask: np.ndarray, min_area=30, max_gap=8):
        """
        mask: (H, W), 0/255
        return: list of region masks (H, W)
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comps = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            comps.append({
                "label": i,
                "bbox": (x, y, x + w, y + h),
                "size": max(w, h),
            })
            
        def bbox_gap(b1, b2):
            x1, y1, x2, y2 = b1
            x3, y3, x4, y4 = b2
            dx = max(x3 - x2, x1 - x4, 0)
            dy = max(y3 - y2, y1 - y4, 0)
            return max(dx, dy)

        groups = []
        visited = set()
        for i in range(len(comps)):
            if i in visited:
                continue
            queue = [i]
            visited.add(i)
            group = [comps[i]]
            while queue:
                cur = queue.pop(0)
                for j in range(len(comps)):
                    if j in visited:
                        continue
                    gap = bbox_gap(comps[cur]["bbox"], comps[j]["bbox"])
                    thresh = max(max_gap, int(0.2 * min(comps[cur]["size"], comps[j]["size"])))
                    if gap <= thresh:
                        visited.add(j)
                        queue.append(j)
                        group.append(comps[j])
            groups.append(group)

        regions = []
        for group in groups:
            region_mask = np.zeros_like(mask)
            for comp in group:
                region_mask[labels == comp["label"]] = 255
            regions.append(region_mask)
        return regions
    
    def _resize(self, image: Image.Image, mask: Image.Image, target_size=256):
        original_w, original_h = image.size
        if original_w > original_h:
            scale = target_size / original_w
            new_w = target_size
            new_h = int(original_h * scale)
        else:
            scale = target_size / original_h
            new_h = target_size
            new_w = int(original_w * scale)
        
        image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        mask_resized = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        return image_resized, mask_resized

    def inpaint(self, image_path: str, mask: np.ndarray, iterations=3, dilate_num: int=2):
        """
        mask: np.ndarray [H,W], int, 0/255
        """
        image = self._load_image(image_path)  # [H, W, C] 0~255 int

        regions = self.split_mask_to_regions(mask)
        current_image = image.copy()
        output = current_image
        for region_mask in regions:
            if dilate_num > 0:
                dilate_value = (2 * dilate_num + 1, 2 * dilate_num + 1)
                kernel = np.ones(dilate_value, np.uint8)
                region_mask = cv2.dilate(region_mask, kernel)

            boxes = self.boxes_from_mask(region_mask)
            crop_result = []
            for box in boxes:
                crop_img, crop_mask, crop_box = self._crop_box(current_image, region_mask, box)
                crop_img = Image.fromarray(crop_img).convert("RGB")
                crop_mask = Image.fromarray(crop_mask).convert("L")
                ori_w, ori_h = crop_img.size
                need_resize = False
                if ori_w > 256 or ori_h > 256:
                    image_resized, mask_resized = self._resize(crop_img, crop_mask, target_size=256)
                    need_resize = True
                else:
                    image_resized, mask_resized = crop_img, crop_mask
                img_pad = self._pad(image_resized)
                mask_pad = self._pad(mask_resized)
                img = np.array(img_pad) / 255.
                img = img[np.newaxis, ...].astype(np.float32)
                img = np.transpose(img, (0, 3, 1, 2)) * 2. - 1.  # [1, C, H, W] -1~1.0
                mask = (np.array(mask_pad) > 0).astype(np.float32)
                mask = mask[np.newaxis, np.newaxis, ...]
                if self._tile_has_watermark(mask):
                    residual_input = img * (1 - mask)
                    output = img
                    for _ in range(iterations):
                        residual_out = self.session.run(
                            [self.output_name],
                            {
                                self.image_input_name: residual_input,
                                self.mask_input_name: mask
                            }
                        )[0]
                        g_imgs = residual_input + residual_out
                        g_imgs = g_imgs * mask + img * (1 - mask)
                        residual_input = g_imgs
                        output = g_imgs
                else:
                    output = img
                crop_image = ((np.transpose(output[0], (1, 2, 0)).clip(-1, 1) + 1.) / 2. * 255.).astype(np.uint8)
                if need_resize:
                    crop_image = crop_image[:image_resized.size[1], :image_resized.size[0], :]
                    crop_image = cv2.resize(crop_image, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
                else:
                    crop_image = crop_image[:ori_h, :ori_w, :]
                crop_result.append((crop_image, crop_box))
            
            output = current_image
            for crop_image, _crop_box in crop_result:
                x1, y1, x2, y2 = _crop_box
                output[y1:y2, x1:x2, :] = crop_image
            current_image = output
        return output # [H, W, C] 0~255 uint8
    

if __name__ == "__main__":
    import os
    input_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "image.jpg")
    mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "mask.png")
    inpaint = GRIGInpaint()
    inpaint.prepare(onnx_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "OnnxModels", "grig_inpaint.encmodel"))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    result = inpaint.inpaint(input_image, mask=mask)

    image = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out.png", image)