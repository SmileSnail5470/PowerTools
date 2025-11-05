from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
from typing import Tuple, Union

Position = Union[str, Tuple[int, int]]

def _calc_position(base_size: Tuple[int,int], overlay_size: Tuple[int,int], position: Position, margin=10):
    bw, bh = base_size
    ow, oh = overlay_size

    if isinstance(position, tuple):
        return position

    pos = position.lower()
    x = y = margin

    if 'left' in pos:
        x = margin
    elif 'right' in pos:
        x = bw - ow - margin
    else:  # center horizontally
        x = (bw - ow) // 2

    if 'top' in pos:
        y = margin
    elif 'bottom' in pos:
        y = bh - oh - margin
    else:  # center vertically
        y = (bh - oh) // 2

    return int(x), int(y)


def add_text_watermark(
    input_image_path: str,
    output_image_path: str,
    text: str,
    font_path: str = None,
    font_size: int = 36,
    color: Tuple[int,int,int,int] = (255,255,255,128),
    position: Position = "bottom-right",
    rotation: float = 0.0,
    scale: float = 1.0,
    margin: int = 10,
    max_width_ratio: float = 0.8,
):
    """在图片上添加文字水印。
        color: (R,G,B,A) A in 0-255
        position: "top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right" 或 (x,y)
        rotation: 顺时针角度 (度)
        scale: 文字相对大小缩放（乘以 font_size）
        max_width_ratio: 当文本过宽时自动换行，最大宽度占原图宽度比例
    """
    im = Image.open(input_image_path).convert("RGBA")
    iw, ih = im.size

    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, int(font_size * scale))
    else:
        font = ImageFont.load_default()

    # 创建绘制层（RGBA）
    txt_layer = Image.new("RGBA", im.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt_layer)

    # 若文本很长，尝试换行以适配宽度
    max_w = int(iw * max_width_ratio)
    lines = [text]
    if font and isinstance(font, ImageFont.FreeTypeFont):
        # 手动换行（简单 greedy）
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            tw, th = draw.textsize(test, font=font)
            if tw > max_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)
    else:
        # 对默认字体不换行
        lines = [text]

    # 计算文本块尺寸
    line_heights = []
    max_line_w = 0
    for line in lines:
        lw, lh = draw.textsize(line, font=font)
        line_heights.append(lh)
        if lw > max_line_w:
            max_line_w = lw
    total_h = sum(line_heights) + (len(lines)-1) * int(0.2 * font_size)

    text_img = Image.new("RGBA", (max_line_w, total_h), (255,255,255,0))
    td = ImageDraw.Draw(text_img)

    y_offset = 0
    for i, line in enumerate(lines):
        td.text((0, y_offset), line, font=font, fill=color)
        y_offset += line_heights[i] + int(0.2 * font_size)

    # 旋转（如果需要）
    if rotation:
        text_img = text_img.rotate(rotation, expand=True)

    # 合并前计算位置
    tx, ty = _calc_position((iw, ih), text_img.size, position, margin=margin)

    # 将文字层粘贴到 txt_layer
    txt_layer.paste(text_img, (tx, ty), text_img)

    # 合并到原图
    out = Image.alpha_composite(im, txt_layer)
    # 如果原图不是需要保留 alpha 的格式，可以转换回 RGB
    if output_image_path.lower().endswith(('.jpg', '.jpeg')):
        out = out.convert("RGB")
    out.save(output_image_path)
    return output_image_path


def add_image_watermark(
    input_image_path: str,
    watermark_image_path: str,
    output_image_path: str,
    position: Position = "bottom-right",
    opacity: float = 0.5,
    rotation: float = 0.0,
    scale: float = 1.0,
    margin: int = 10,
    relative_to: str = "base"  # "base" (scale relative to base image width) or "watermark" (scale relative to watermark)
):
    """
    在图片上添加图片水印。
    opacity: 0.0 - 1.0
    scale: 相对缩放因子。如果 relative_to == "base"，则 watermark_width = base_width * scale。
           如果 == "watermark"，则 watermark_size = original_wm_size * scale。
    position: 支持同 add_text_watermark
    """
    base = Image.open(input_image_path).convert("RGBA")
    wm = Image.open(watermark_image_path).convert("RGBA")
    bw, bh = base.size
    ww, wh = wm.size

    # 缩放
    if relative_to == "base":
        target_w = int(bw * scale)
        # 保持纵横比
        if ww != 0:
            target_h = int((target_w / ww) * wh)
        else:
            target_h = wh
        wm = wm.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)
    else:  # relative_to == "watermark"
        target_w = int(ww * scale)
        target_h = int(wh * scale)
        wm = wm.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)

    # 旋转
    if rotation:
        wm = wm.rotate(rotation, expand=True)

    # 调整透明度
    if opacity < 0: opacity = 0.0
    if opacity > 1: opacity = 1.0
    if opacity < 1:
        # 给 wm 的 alpha 通道乘以 opacity
        alpha = wm.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        wm.putalpha(alpha)

    # 计算位置
    x, y = _calc_position((bw, bh), wm.size, position, margin=margin)

    # 生成图层并粘贴
    layer = Image.new("RGBA", base.size, (255,255,255,0))
    layer.paste(wm, (x, y), wm)
    out = Image.alpha_composite(base, layer)

    if output_image_path.lower().endswith(('.jpg', '.jpeg')):
        out = out.convert("RGB")
    out.save(output_image_path)
    return output_image_path


# ---- 示例用法 ----
if __name__ == "__main__":
    # 文字水印示例
    add_text_watermark(
        "input.jpg",
        "out_text.jpg",
        text="示例水印 © 2025",
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # 请按实际系统路径替换
        font_size=48,
        color=(255,255,255,160),
        position="bottom-right",
        rotation=-30,
        scale=1.0,
        margin=20,
    )

    # 图片水印示例
    add_image_watermark(
        "input.jpg",
        "logo.png",
        "out_image.jpg",
        position="top-right",
        opacity=0.6,
        rotation=15,
        scale=0.2,  # 如果 relative_to="base"，此处为基于原图宽度的比例
        relative_to="base",
        margin=30,
    )