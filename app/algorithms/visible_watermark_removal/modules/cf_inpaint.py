import os
import math
import platform
import sys
import cv2
import numpy as np
import onnxruntime as ort
ort.preload_dlls(directory="")
from PIL import Image
from app.algorithms import general_inference_session


class CoordFillInpaint():
    def __init__(self):
        pass

    def _hash_cuda_gpu(self):
        if platform.system() != "Windows":
            return True
        cuda_path = r"C:\Program Files\NVIDIA Corporation"
        if os.path.exists(cuda_path):
            return True
        return False

    def _create_predictor(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        elif "CUDAExecutionProvider" in available and self._hash_cuda_gpu():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
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

    def _resize_fn(self, img, size):
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW → HWC
                img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img.astype(np.uint8))
        elif isinstance(img, Image.Image):
            pass
        else:
            raise TypeError("Unsupported image type")

        img = img.resize(size, Image.BILINEAR)

        img = np.asarray(img).astype(np.float32) / 255.0  # HWC
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img
    
    def _to_mask(self, mask):
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[0] == 3:
                mask = np.transpose(mask, (1, 2, 0))
            if mask.ndim == 3:
                mask = Image.fromarray(mask.astype(np.uint8)).convert("L")
            elif mask.ndim == 2:
                mask = Image.fromarray(mask.astype(np.uint8))
            else:
                raise ValueError("Unsupported mask ndarray shape")
        elif isinstance(mask, Image.Image):
            mask = mask.convert("L")
        else:
            raise TypeError("Unsupported mask type")
        
        mask = np.asarray(mask).astype(np.float32)
        mask = mask[np.newaxis, ...]  # (1,H,W)
        return mask

    def _load_image(self, image: np.ndarray, mask: np.ndarray, block_size=512, fill_color=(0, 0, 0)):
        img = Image.fromarray(image).convert("RGB")
        mask = Image.fromarray(mask).convert("RGB")
        assert img.size == mask.size, "image and mask size do not match"
        w, h = img.size

        cols = math.ceil(w / block_size)
        rows = math.ceil(h / block_size)

        new_w = cols * block_size
        new_h = rows * block_size

        canvas = Image.new("RGB", (new_w, new_h), fill_color)
        canvas.paste(img, (0, 0))

        canvas_mask = Image.new("RGB", (new_w, new_h), fill_color)
        canvas_mask.paste(mask, (0, 0))

        tiles = {"image": [], "mask": []}
        for r in range(rows):
            for c in range(cols):
                left = c * block_size
                upper = r * block_size
                right = left + block_size
                lower = upper + block_size
                tile = canvas.crop((left, upper, right, lower))
                tiles['image'].append(tile)
                tile_mask = canvas_mask.crop((left, upper, right, lower))
                tiles['mask'].append(tile_mask)
        return tiles
    
    def _merge_image(self, tiles, orig_size, block_size=512):
        orig_w, orig_h = orig_size
        cols = math.ceil(orig_w / block_size)
        rows = math.ceil(orig_h / block_size)

        assert len(tiles) == cols * rows, "tiles number does not match"

        full_w = cols * block_size
        full_h = rows * block_size
        canvas = Image.new("RGB", (full_w, full_h))

        idx = 0
        for r in range(rows):
            for c in range(cols):
                x = c * block_size
                y = r * block_size
                canvas.paste(tiles[idx], (x, y))
                idx += 1
        result = canvas.crop((0, 0, orig_w, orig_h))
        return np.array(result)
    
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
    
    def forward(self, image: np.ndarray, mask: np.ndarray):
        """
        image: [H, W, C] RGB 0~255
        mask:  [H,W], int8, 0~255
        """
        input_tiles = self._load_image(image, mask)

        output_titles = []
        for img, mask in zip(input_tiles["image"], input_tiles["mask"]):
            input_image = self._resize_fn(img=img, size=(512, 512))
            input_image = (input_image - 0.5) / 0.5
            input_mask = self._resize_fn(img=mask, size=(512, 512))
            input_mask = self._to_mask(input_mask)
            input_mask[input_mask > 0] = 1
            input_mask = 1 - input_mask

            input_mask = input_mask[np.newaxis, ...]
            input_image = input_image[np.newaxis, ...]
            if self._tile_has_watermark(input_mask):
                out_tile = self.session.run(
                    [self.output_name],
                    {
                        self.image_input_name: input_image,
                        self.mask_input_name: input_mask
                    }
                )[0]
            else:
                out_tile = input_image[0]
            pred = out_tile  # (3,512,512)
            pred = np.transpose(pred, (1, 2, 0))  # (512,512,3)
            pred = (pred * 255.0).astype(np.uint8)
            output_image = Image.fromarray(pred).convert("RGB")
            output_titles.append(output_image)

        out = self._merge_image(output_titles, (image.shape[1], image.shape[0]))
        return out
    
    def _crop_box(self, image, mask, box):
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

        w = box_w + 128 * 2
        h = box_h + 128 * 2

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

    def inpaint(self, image_path: str, mask: np.ndarray, enable_crop: bool=True, dilate_num: int=2):
        """
        mask: np.ndarray [H,W], int8, 0~255
        """
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        if enable_crop:
            regions = self.split_mask_to_regions(mask)
            current_image = img.copy()
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
                    crop_image = self.forward(image=crop_img, mask=crop_mask)  # (1, C, H, W) 0~255 float
                    crop_result.append((crop_image, crop_box))
                
                output = current_image
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    output[y1:y2, x1:x2, :] = crop_image
                current_image = output
        else:
            output = self.forward(image=img, mask=mask)

        return output  # [H, W, C] 0~255 float
    


if __name__ == "__main__":
    import os
    input_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "image.jpg")
    mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "mask.png")
    inpaint = CoordFillInpaint()
    inpaint.prepare(onnx_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "OnnxModels", "cf.onnx"))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
    # mask 是 (H, W) 值是 0/255
    mask = ((mask > 0) * 255).astype(np.uint8)

    result = inpaint.inpaint(input_image, mask=mask)

    image = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out.png", image)
