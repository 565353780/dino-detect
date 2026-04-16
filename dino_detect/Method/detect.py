import os
import math
import cv2
import torch
import numpy as np
from typing import List, Union
from torchvision.transforms import v2


def createTransform() -> v2.Compose:
    return v2.Compose([
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


@torch.no_grad()
def preprocessImages(
    image_tensor: torch.Tensor,
    transform: v2.Compose,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Preprocess image tensor for DINO-family models.

    Args:
        image_tensor: [B, H, W, 3], float32, range [0, 1]
        transform: normalization transform
        device: target device
        dtype: target dtype

    Returns:
        (processed_tensor, input_dtype, input_device)
    """
    input_dtype = image_tensor.dtype
    input_device = image_tensor.device

    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor.to(device, dtype=dtype)
    image_tensor = transform(image_tensor)

    return image_tensor, input_dtype, input_device


@torch.no_grad()
def postprocessFeatures(
    x_norm: torch.Tensor,
    input_device: torch.device,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    return x_norm.to(input_device, dtype=input_dtype)


@torch.no_grad()
def detectFile(
    detector,
    image_file_path: str,
) -> Union[torch.Tensor, None]:
    if not os.path.exists(image_file_path):
        print('[ERROR][detectFile]')
        print('\t image file not exist!')
        print('\t image_file_path:', image_file_path)
        return None

    image_bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print('[ERROR][detectFile]')
        print('\t failed to read image!')
        print('\t image_file_path:', image_file_path)
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).unsqueeze(0)

    return detector.detect(image_tensor)


@torch.no_grad()
def toPCAImages(dino_feats: torch.Tensor) -> List[np.ndarray]:
    """Project DINO patch features to RGB via SVD-based PCA.

    Args:
        dino_feats: (N, T, C) float tensor — raw patch features for N views.
                    T must be a perfect square (e.g. 1024 = 32x32).
    Returns:
        List of N uint8 BGR images at patch-grid resolution (h, w, 3).
    """
    N, T, C = dino_feats.shape
    h = w = int(math.sqrt(T))
    assert h * w == T, f"Non-square token grid (T={T})"

    flat = dino_feats.reshape(N * T, C).float()
    mean = flat.mean(dim=0, keepdim=True)
    X = flat - mean

    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    proj = X @ Vh[:3].T

    cmin = proj.min(dim=0).values
    cmax = proj.max(dim=0).values
    proj = (proj - cmin) / (cmax - cmin).clamp(min=1e-6)
    proj = proj.reshape(N, h, w, 3).cpu().numpy()

    images = []
    for i in range(N):
        img = (proj[i] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images
