"""
ONNX-compatible utility operations for RGVI inference.
Implements numpy equivalents of PyTorch operations used in the pipeline.
Completely torch-free.
"""

import numpy as np
from scipy.ndimage import zoom


def grid_sample_np(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    NumPy implementation of torch.nn.functional.grid_sample.
    
    Args:
        input: numpy array of shape (N, C, H_in, W_in)
        grid: numpy array of shape (N, H_out, W_out, 2) with values in [-1, 1]
        mode: 'bilinear' or 'nearest'
        padding_mode: 'zeros', 'border', or 'reflection'
        align_corners: bool, same as torch's align_corners
    
    Returns:
        output: numpy array of shape (N, C, H_out, W_out)
    """
    N, C, H_in, W_in = input.shape
    N_grid, H_out, W_out, _ = grid.shape
    assert N == N_grid, "Batch sizes must match"

    output = np.zeros((N, C, H_out, W_out), dtype=input.dtype)

    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                # Get normalized coordinates
                gx = grid[n, h, w, 0]  # x (width dimension)
                gy = grid[n, h, w, 1]  # y (height dimension)

                if align_corners:
                    # Map from [-1, 1] to [0, H_in-1] and [0, W_in-1]
                    ix = ((gx + 1) / 2) * (W_in - 1)
                    iy = ((gy + 1) / 2) * (H_in - 1)
                else:
                    # Map from [-1, 1] to [-0.5, H_in-0.5] and [-0.5, W_in-0.5]
                    ix = ((gx + 1) / 2) * W_in - 0.5
                    iy = ((gy + 1) / 2) * H_in - 0.5

                if mode == 'bilinear':
                    # Get integer coordinates of the 4 surrounding pixels
                    ix_nw = int(np.floor(ix))
                    iy_nw = int(np.floor(iy))
                    ix_ne = ix_nw + 1
                    iy_ne = iy_nw
                    ix_sw = ix_nw
                    iy_sw = iy_nw + 1
                    ix_se = ix_nw + 1
                    iy_se = iy_nw + 1

                    # Compute weights
                    wx = ix - ix_nw
                    wy = iy - iy_nw

                    def get_pixel_bounded(x, y, padding_mode='zeros'):
                        """Get pixel value with boundary handling."""
                        if padding_mode == 'zeros':
                            if x < 0 or x >= W_in or y < 0 or y >= H_in:
                                return 0.0
                            return input[n, :, y, x]
                        elif padding_mode == 'border':
                            x = max(0, min(x, W_in - 1))
                            y = max(0, min(y, H_in - 1))
                            return input[n, :, int(y), int(x)]
                        elif padding_mode == 'reflection':
                            # Reflect coordinates
                            if x < 0:
                                x = -x - 1
                            if x >= W_in:
                                x = 2 * W_in - x - 1
                            if y < 0:
                                y = -y - 1
                            if y >= H_in:
                                y = 2 * H_in - y - 1
                            return input[n, :, int(y), int(x)]
                        else:
                            return 0.0

                    # Bilinear interpolation
                    val = (1 - wx) * (1 - wy) * get_pixel_bounded(ix_nw, iy_nw, padding_mode) \
                        + wx * (1 - wy) * get_pixel_bounded(ix_ne, iy_ne, padding_mode) \
                        + (1 - wx) * wy * get_pixel_bounded(ix_sw, iy_sw, padding_mode) \
                        + wx * wy * get_pixel_bounded(ix_se, iy_se, padding_mode)

                    output[n, :, h, w] = val

                elif mode == 'nearest':
                    # Nearest neighbor
                    ix_n = int(np.round(ix))
                    iy_n = int(np.round(iy))
                    
                    if padding_mode == 'zeros':
                        if ix_n < 0 or ix_n >= W_in or iy_n < 0 or iy_n >= H_in:
                            output[n, :, h, w] = 0.0
                        else:
                            output[n, :, h, w] = input[n, :, iy_n, ix_n]
                    elif padding_mode == 'border':
                        ix_n = max(0, min(ix_n, W_in - 1))
                        iy_n = max(0, min(iy_n, H_in - 1))
                        output[n, :, h, w] = input[n, :, iy_n, ix_n]
                    else:
                        output[n, :, h, w] = input[n, :, iy_n, ix_n]

    return output


def backward_warp(x, flow):
    """
    NumPy implementation of backward_warp using grid_sample.
    
    Args:
        x: numpy array of shape (B, C, H, W) - input image/feature
        flow: numpy array of shape (B, 2, H, W) - optical flow
    
    Returns:
        warped: numpy array of shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    
    # Create grid
    grid_h = np.arange(0, H).reshape(1, H, 1).repeat(B, axis=0).repeat(W, axis=2)
    grid_w = np.arange(0, W).reshape(1, 1, W).repeat(B, axis=0).repeat(H, axis=1)
    grid = np.stack([grid_w, grid_h], axis=3).astype(x.dtype)
    
    # Add flow to grid
    flow_permuted = flow.transpose(0, 2, 3, 1)  # (B, H, W, 2)
    grid_flow = grid + flow_permuted
    
    # Normalize to [-1, 1] for grid_sample
    grid_flow_w = 2 * grid_flow[:, :, :, 0] / (W - 1) - 1
    grid_flow_h = 2 * grid_flow[:, :, :, 1] / (H - 1) - 1
    norm_grid_flow = np.stack([grid_flow_w, grid_flow_h], axis=3)
    
    return grid_sample_np(x, norm_grid_flow, align_corners=True)


def _bicubic_kernel(x):
    """Cubic convolution kernel (a=-0.5 for bicubic)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    
    mask1 = x <= 1
    result[mask1] = 1.5 * x[mask1] ** 3 - 2.5 * x[mask1] ** 2 + 1
    
    mask2 = (x > 1) & (x <= 2)
    result[mask2] = -0.5 * x[mask2] ** 3 + 2.5 * x[mask2] ** 2 - 4 * x[mask2] + 2
    
    return result


def resize_bicubic_np(img, new_height, new_width):
    """
    Bicubic resize for a single image or batched images.
    
    Args:
        img: numpy array of shape (H, W, C) or (B, C, H, W)
        new_height: target height
        new_width: target width
    
    Returns:
        resized: numpy array
    """
    if img.ndim == 3:
        # Single image (H, W, C)
        H, W, C = img.shape
        result = np.zeros((new_height, new_width, C), dtype=img.dtype)
        
        scale_y = H / new_height
        scale_x = W / new_width
        
        for y in range(new_height):
            src_y = (y + 0.5) * scale_y - 0.5
            y_int = int(np.floor(src_y))
            y_frac = src_y - y_int
            
            for x in range(new_width):
                src_x = (x + 0.5) * scale_x - 0.5
                x_int = int(np.floor(src_x))
                x_frac = src_x - x_int
                
                # Gather 4x4 neighborhood
                for c in range(C):
                    val = 0.0
                    for dy in range(-1, 3):
                        ky = _bicubic_kernel(dy - y_frac)
                        for dx in range(-1, 3):
                            kx = _bicubic_kernel(dx - x_frac)
                            sy = y_int + dy
                            sx = x_int + dx
                            if 0 <= sy < H and 0 <= sx < W:
                                val += img[sy, sx, c] * ky * kx
                    result[y, x, c] = val
        return result
    
    elif img.ndim == 4:
        # Batched (B, C, H, W)
        B, C, H, W = img.shape
        # Transpose to (B, H, W, C) for processing
        img_t = img.transpose(0, 2, 3, 1)
        result = np.zeros((B, new_height, new_width, C), dtype=img.dtype)
        
        scale_y = H / new_height
        scale_x = W / new_width
        
        for b in range(B):
            for y in range(new_height):
                src_y = (y + 0.5) * scale_y - 0.5
                y_int = int(np.floor(src_y))
                y_frac = src_y - y_int
                
                for x in range(new_width):
                    src_x = (x + 0.5) * scale_x - 0.5
                    x_int = int(np.floor(src_x))
                    x_frac = src_x - x_int
                    
                    for c in range(C):
                        val = 0.0
                        for dy in range(-1, 3):
                            ky = _bicubic_kernel(dy - y_frac)
                            for dx in range(-1, 3):
                                kx = _bicubic_kernel(dx - x_frac)
                                sy = y_int + dy
                                sx = x_int + dx
                                if 0 <= sy < H and 0 <= sx < W:
                                    val += img_t[b, sy, sx, c] * ky * kx
                        result[b, y, x, c] = val
        
        return result.transpose(0, 3, 1, 2)
    
    else:
        raise ValueError(f"Unsupported shape: {img.shape}")


def interpolate_np(x, size=None, scale_factor=None, mode='bilinear', align_corners=True):
    """
    NumPy implementation of torch.nn.functional.interpolate.
    
    Args:
        x: numpy array (N, C, H, W)
        size: (new_h, new_w) tuple
        scale_factor: scaling factor (ignored if size is provided)
        mode: 'bilinear', 'bicubic', or 'nearest'
        align_corners: bool
    
    Returns:
        resized: numpy array (N, C, new_h, new_w)
    """
    if size is not None:
        new_h, new_w = size
    elif scale_factor is not None:
        new_h = int(round(x.shape[2] * scale_factor))
        new_w = int(round(x.shape[3] * scale_factor))
    else:
        return x.copy()
    
    N, C, H, W = x.shape
    
    if mode == 'bilinear':
        # Simple bilinear interpolation using grid_sample
        # Create normalized grid
        grid_y = np.linspace(-1, 1, new_h) if align_corners else np.linspace(-1 + 1/new_h, 1 - 1/new_h, new_h)
        grid_x = np.linspace(-1, 1, new_w) if align_corners else np.linspace(-1 + 1/new_w, 1 - 1/new_w, new_w)
        gy, gx = np.meshgrid(grid_y, grid_x, indexing='ij')
        grid = np.stack([gx, gy], axis=-1)  # (new_h, new_w, 2)
        grid = np.tile(grid[None, :, :, :], (N, 1, 1, 1))  # (N, new_h, new_w, 2)
        
        return grid_sample_np(x, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
    
    elif mode == 'bicubic':
        return resize_bicubic_np(x, new_h, new_w)
    
    elif mode == 'nearest':
        grid_y = np.linspace(-1, 1, new_h) if align_corners else np.linspace(-1 + 1/new_h, 1 - 1/new_h, new_h)
        grid_x = np.linspace(-1, 1, new_w) if align_corners else np.linspace(-1 + 1/new_w, 1 - 1/new_w, new_w)
        gy, gx = np.meshgrid(grid_y, grid_x, indexing='ij')
        grid = np.stack([gx, gy], axis=-1)
        grid = np.tile(grid[None, :, :, :], (N, 1, 1, 1))
        
        return grid_sample_np(x, grid, mode='nearest', padding_mode='zeros', align_corners=align_corners)
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def interpolate_nearest_np(x, size=None, scale_factor=None):
    """Nearest-neighbor interpolation."""
    return interpolate_np(x, size=size, scale_factor=scale_factor, mode='nearest')


def avg_pool2d_np(x, kernel_size, stride=1, padding=0):
    """
    NumPy implementation of torch.nn.functional.avg_pool2d.
    
    Args:
        x: numpy array (N, C, H, W)
        kernel_size: int or tuple
        stride: int or tuple
        padding: int or tuple
    
    Returns:
        pooled: numpy array (N, C, H_out, W_out)
    """
    if isinstance(kernel_size, int):
        k_h = k_w = kernel_size
    else:
        k_h, k_w = kernel_size
    
    if isinstance(stride, int):
        s_h = s_w = stride
    else:
        s_h, s_w = stride
    
    if isinstance(padding, int):
        p_h = p_w = padding
    else:
        p_h, p_w = padding
    
    N, C, H, W = x.shape
    
    # Apply padding
    if p_h > 0 or p_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant', constant_values=0)
    
    H_pad, W_pad = x.shape[2], x.shape[3]
    
    H_out = (H_pad - k_h) // s_h + 1
    W_out = (W_pad - k_w) // s_w + 1
    
    output = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * s_h
                    w_start = w * s_w
                    output[n, c, h, w] = np.mean(
                        x[n, c, h_start:h_start + k_h, w_start:w_start + k_w]
                    )
    
    return output


def masks_to_boxes_np(mask):
    """
    NumPy implementation of torchvision.ops.masks_to_boxes.
    
    Args:
        mask: numpy array (1, H, W) - binary mask
    
    Returns:
        bbox: numpy array (1, 4) - [x1, y1, x2, y2]
    """
    mask_bool = mask[0] > 0
    
    # Find non-zero pixels
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return np.array([[0, 0, 0, 0]], dtype=np.float32)
    
    y_min = np.argmax(rows)
    y_max = len(rows) - np.argmax(rows[::-1]) - 1
    x_min = np.argmax(cols)
    x_max = len(cols) - np.argmax(cols[::-1]) - 1
    
    return np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
