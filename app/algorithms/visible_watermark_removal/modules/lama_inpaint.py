import platform
import sys
import cv2
import numpy as np
import onnxruntime as ort
ort.preload_dlls(directory="")
from PIL import Image


class LamaInpaint():
    def __init__(self):
        self.tile_size = 512

    def _create_predictor(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        if is_apple_silicon:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        self.session = ort.InferenceSession(
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

    def get_image(self, image, need_norm=True):
        if isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise Exception("Input image should be either PIL Image or numpy array!")

        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # chw
        elif img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 4:
            img = img[0]

        if need_norm:
            img = img.astype(np.float32) / 255

        return img


    def ceil_modulo(self, x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod


    def scale_image(self, img, factor, interpolation=cv2.INTER_AREA):
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))

        img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

        if img.ndim == 2:
            img = img[None, ...]
        else:
            img = np.transpose(img, (2, 0, 1))
        return img


    def pad_img_to_modulo(self, img, mod):
        channels, height, width = img.shape
        out_height = self.ceil_modulo(height, mod)
        out_width = self.ceil_modulo(width, mod)
        return np.pad(
            img,
            ((0, 0), (0, out_height - height), (0, out_width - width)),
            mode="symmetric",
        )


    def prepare_img_and_mask(self, image, mask, scale_factor=None):
        out_image = self.get_image(image)
        out_mask = self.get_image(mask)
        pad_out_to_modulo = self.tile_size

        if scale_factor is not None:
            out_image = self.scale_image(out_image, scale_factor)
            out_mask = self.scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

        if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
            out_image = self.pad_img_to_modulo(out_image, pad_out_to_modulo)
            out_mask = self.pad_img_to_modulo(out_mask, pad_out_to_modulo)

        out_image = out_image[np.newaxis, ...]
        out_mask = out_mask[np.newaxis, ...]

        return out_image, out_mask

    def open_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        return image
    
    def tile_has_watermark(self, mask_tile, bin_thresh=0.5, min_area_ratio=0.001, min_area_abs=300, min_sum=80, max_gap=8):
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
    
    def forward(self, image, mask):
        """
        image: (H, W, C)
        mask: (H, W)
        """
        H_orig, W_orig = image.shape[:2]
        image, mask = self.prepare_img_and_mask(image, mask)
        _, _, H_pad, W_pad = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        for y in range(0, H_pad, self.tile_size):
            for x in range(0, W_pad, self.tile_size):
                img_tile = image[:, :, y:y+self.tile_size, x:x+self.tile_size]
                mask_tile = mask[:, :, y:y+self.tile_size, x:x+self.tile_size]
                if self.tile_has_watermark(mask_tile=mask_tile):
                    out_tile = self.session.run(
                        [self.output_name],
                        {
                            self.image_input_name: img_tile,
                            self.mask_input_name: mask_tile
                        }
                    )[0]
                else:
                    out_tile = img_tile * 255.0
                output[:, :, y:y+self.tile_size, x:x+self.tile_size] = out_tile
        output = output[:, :, :H_orig, :W_orig]
        return output

    def inpaint(self, image_path: str, mask: np.ndarray, enable_crop: bool=True, dilate_num: int=2):
        """
        mask: np.ndarray [H,W], int, 0/255
        """
        image = self.open_image(image_path)
        if enable_crop:
            regions = self.split_mask_to_regions(mask)
            current_image = image.copy()
            output = np.transpose(current_image, (2, 0, 1))
            output = output[np.newaxis, ...]
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
                
                output = np.transpose(current_image, (2, 0, 1))
                output = output[np.newaxis, ...]
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    output[:, :, y1:y2, x1:x2] = crop_image
                current_image = np.transpose(output[0], (1, 2, 0))
        else:
            output = self.forward(image=image, mask=mask)
        return output / 255.0  # (1, C, H, W) 0~1 float
    

if __name__ == "__main__":
    import os
    input_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "image.jpg")
    mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assess", "mask.png")
    inpaint = LamaInpaint()
    inpaint.prepare(onnx_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "OnnxModels", "lama_fp32.onnx"))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = ((mask > 0) * 255).astype(np.uint8)

    result = inpaint.inpaint(input_image, mask=mask)

    res = (np.transpose(result[0], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)

    image = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out.png", image)