import os
import cv2
import torch
import numpy as np
from typing import Union
from torchvision.transforms import v2

from dino_detect.Model.vision_transformer import (
    vit_large,
    vit_base,
    vit_small,
    vit_huge2,
    vit_7b,
)

DINOV3_COMMON_KWARGS = {
    'patch_size': 16,
    'n_storage_tokens': 4,
    'norm_layer': 'layernormbf16',
    'layerscale_init': 1e-5,
    'mask_k_bias': True,
    'pos_embed_rope_rescale_coords': 2,
}


class Detector(object):
    def __init__(
        self,
        model_type: str,
        model_file_path: Union[str, None]=None,
        dtype = 'auto',
        device: str = 'cpu',
    ) -> None:
        self.device = device
        if dtype == 'auto':
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = dtype

        model_configs = {
            'small': {
                'factory': vit_small,
                'kwargs': {
                    **DINOV3_COMMON_KWARGS,
                    'ffn_layer': 'mlp',
                },
            },
            'small+': {
                'factory': vit_small,
                'kwargs': {
                    **DINOV3_COMMON_KWARGS,
                    'ffn_ratio': 6,
                    'ffn_layer': 'swiglu',
                },
            },
            'base': {
                'factory': vit_base,
                'kwargs': {
                    **DINOV3_COMMON_KWARGS,
                    'ffn_layer': 'mlp',
                },
            },
            'large': {
                'factory': vit_large,
                'kwargs': {
                    **DINOV3_COMMON_KWARGS,
                    'ffn_layer': 'mlp',
                },
            },
            'large+': {
                'factory': vit_large,
                'kwargs': {
                    **DINOV3_COMMON_KWARGS,
                    'ffn_ratio': 6,
                    'ffn_layer': 'swiglu',
                },
            },
            'huge+': {
                'factory': vit_huge2,
                'kwargs': {
                    **DINOV3_COMMON_KWARGS,
                    'ffn_ratio': 6,
                    'ffn_layer': 'swiglu',
                },
            },
            '7b': {
                'factory': vit_7b,
                'kwargs': {
                    **DINOV3_COMMON_KWARGS,
                    'qkv_bias': False,
                    'ffn_layer': 'swiglu64',
                },
            },
        }

        if model_type not in model_configs:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {list(model_configs.keys())}"
            )

        config = model_configs[model_type]
        self.model = config['factory'](**config['kwargs'])

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        self.transform = v2.Compose([
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.is_valid = False
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            self.is_valid = False
            return False

        model_state_dict = torch.load(model_file_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(model_state_dict, strict=True)

        print('[INFO][Detector::loadModel]')
        print('\t model loaded from:', model_file_path)
        self.is_valid = True
        return True

    @torch.no_grad()
    def detect(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor: [B, H, W, 3], float32, range [0, 1]
        Returns:
            x_norm: [B, N, C], same dtype and device as input
        """
        input_dtype = image_tensor.dtype
        input_device = image_tensor.device

        # [B, H, W, 3] -> [B, 3, H, W]
        image_tensor = image_tensor.permute(0, 3, 1, 2)

        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        image_tensor = self.transform(image_tensor)

        device_type = self.device if isinstance(self.device, str) else self.device.type
        device_type = device_type.split(":")[0]
        with torch.autocast(device_type, dtype=self.dtype):
            dino_features_dict = self.model.forward_features(image_tensor)

        assert isinstance(dino_features_dict, dict)

        cls_token = dino_features_dict["x_norm_clstoken"].unsqueeze(1)
        storage_tokens = dino_features_dict["x_storage_tokens"]
        patch_tokens = dino_features_dict["x_norm_patchtokens"]
        x_norm = torch.cat([cls_token, storage_tokens, patch_tokens], dim=1)

        x_norm = x_norm.to(input_device, dtype=input_dtype)

        return x_norm

    @torch.no_grad()
    def detectFile(self, image_file_path: str) -> Union[torch.Tensor, None]:
        if not os.path.exists(image_file_path):
            print('[ERROR][Detector::detectFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        image_bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print('[ERROR][Detector::detectFile]')
            print('\t failed to read image!')
            print('\t image_file_path:', image_file_path)
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # [H, W, 3] -> [1, H, W, 3], float32, range [0, 1]
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        dino_feature = self.detect(image_tensor)

        return dino_feature
