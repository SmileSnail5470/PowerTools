MEDIA_IMAGE = "image"
MEDIA_VIDEO = "video"
MEDIA_TYPES = (MEDIA_IMAGE, MEDIA_VIDEO)
MEDIA_LABELS = {
    MEDIA_IMAGE: "图片模型",
    MEDIA_VIDEO: "视频模型",
}


MODEL_DIR_SIZES = {
    "cpu": {
        "blind_watermark_addition": 1027,
        "visible_watermark_removal": 2464,
        "segment": 857,
        "ocr": 207,
        "tracker": 119,
        "video_inpainting/ppt": 250,
        "image_edit/general_edit": 5756,
        "image_edit/reverse_edit": 2198,
        "image_edit/sr_edit": 3339,
        "image_restoration/general_restoration": 6802,
    },
    "gpu": {
        "blind_watermark_addition": 1027,
        "visible_watermark_removal": 2369,
        "segment": 1610,
        "ocr": 207,
        "tracker": 119,
        "video_inpainting/ppt": 175,
        "image_edit/general_edit": 5756,
        "image_edit/reverse_edit": 2198,
        "image_edit/sr_edit": 2528,
        "image_restoration/general_restoration": 6640,
    },
}


AI_CAPABILITIES = (
    {
        "key": "watermark_add",
        "title": "水印添加",
        "description": "为图片、视频添加可见水印与不可见盲水印，并支持盲水印提取",
        "cfg_attr": "localWatermarkAddEnabled",
        "media": {
            MEDIA_IMAGE: ["blind_watermark_addition"],
            MEDIA_VIDEO: ["blind_watermark_addition"],
        },
    },
    {
        "key": "watermark_remove",
        "title": "水印移除",
        "description": "智能检测并去除图片、视频中的可见水印、台标与字幕",
        "cfg_attr": "localWatermarkRemoveEnabled",
        "media": {
            MEDIA_IMAGE: ["visible_watermark_removal", "segment", "image_edit/general_edit", "image_restoration/general_restoration"],
            MEDIA_VIDEO: [
                "visible_watermark_removal", "segment", "image_edit/general_edit", "image_restoration/general_restoration", 
                "video_inpainting/ppt", "tracker"
            ],
        },
    },
    {
        "key": "text_extract",
        "title": "文字提取",
        "description": "识别并提取图片中的文字、视频中的字幕内容",
        "cfg_attr": "localTextExtractEnabled",
        "media": {
            MEDIA_IMAGE: ["ocr"],
            MEDIA_VIDEO: ["ocr"],
        },
    },
    {
        "key": "blind_watermark_remove",
        "title": "暗印去除",
        "description": "去除图片、视频中不可见的暗水印（盲水印）痕迹",
        "cfg_attr": "localBlindWatermarkRemoveEnabled",
        "media": {
            MEDIA_IMAGE: ["image_edit/reverse_edit", "image_edit/sr_edit"],
            MEDIA_VIDEO: ["image_edit/reverse_edit", "image_edit/sr_edit"],
        },
    },
    {
        "key": "image_edit",
        "title": "图像编辑",
        "description": "提示词驱动的智能图像编辑、局部重绘与内容替换",
        "cfg_attr": "localImageEditEnabled",
        "media": {
            MEDIA_IMAGE: ["image_edit/general_edit"],
            MEDIA_VIDEO: ["image_edit/general_edit"],
        },
    },
    {
        "key": "image_restoration",
        "title": "图像修复",
        "description": "画质修复、超分辨率、去雾去噪去模糊等画面增强",
        "cfg_attr": "localImageRestorationEnabled",
        "media": {
            MEDIA_IMAGE: ["image_restoration/general_restoration", "image_edit/sr_edit"],
            MEDIA_VIDEO: ["image_restoration/general_restoration", "image_edit/sr_edit"],
        },
    },
)

def capability_keys() -> list:
    return [item["key"] for item in AI_CAPABILITIES]


def get_capability(key: str) -> dict:
    for item in AI_CAPABILITIES:
        if item["key"] == key:
            return item
    return {}


def media_dirs(key: str, media_type: str) -> list:
    return list(get_capability(key).get("media", {}).get(media_type, []))


def capability_dirs(key: str, media_types=None) -> list:
    if media_types is None:
        media_types = MEDIA_TYPES
    dirs = []
    for media_type in MEDIA_TYPES:
        if media_type not in media_types:
            continue
        for item in media_dirs(key, media_type):
            if item not in dirs:
                dirs.append(item)
    return dirs


def default_media_types() -> dict:
    return {media_type: True for media_type in MEDIA_TYPES}


def format_size(total_mb: float, has_unknown: bool = False) -> str:
    if total_mb <= 0:
        return "待定" if has_unknown else "-- MB"
    text = f"{total_mb / 1024:.2f} GB" if total_mb >= 1024 else f"{int(round(total_mb))} MB"
    return f"{text}+" if has_unknown else text


def estimate_size(dirs, variant: str) -> str:
    sizes = MODEL_DIR_SIZES.get(variant, {})
    total = 0.0
    has_unknown = False
    for item in dirs:
        size = sizes.get(item)
        if size is None:
            has_unknown = True
            continue
        total += size
    return format_size(total, has_unknown)
