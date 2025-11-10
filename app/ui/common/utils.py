

def get_file_type(input_file: str):
    images_suffix = ["png", "jpg", "jpeg", "bmp", "avif", "webp"]
    videos_sufffix = ["mp4", "avi", "mov", "mkv"]
    ext = input_file.lower().split(".")[-1]
    if ext in images_suffix:
        return "image"
    if ext in videos_sufffix:
        return "video"
    else:
        return None