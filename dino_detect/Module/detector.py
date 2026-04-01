import os
import torch
from PIL import Image
from typing import Union
from torchvision import transforms

from dino_detect.Model.vision_transformer import (
    vit_giant2,
    vit_large,
    vit_base,
    vit_small,
    vit_so400m,
    vit_huge2,
    vit_7b,
)


class Detector(object):
    def __init__(self,
                 model_type: str,
                 model_file_path: Union[str, None]=None,
                 dtype = 'auto',
                 device: str = 'cpu') -> None:
        self.device = device
        if dtype == 'auto':
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = dtype

        model_configs = {
            'giant2': {
                'factory': vit_giant2,
                'kwargs': {
                    'patch_size': 16,
                    'n_storage_tokens': 4,
                    'ffn_layer': 'swiglu',
                },
            },
            'large': {
                'factory': vit_large,
                'kwargs': {
                    'patch_size': 16,
                    'n_storage_tokens': 4,
                    'ffn_layer': 'mlp',
                },
            },
            'base': {
                'factory': vit_base,
                'kwargs': {
                    'patch_size': 16,
                    'n_storage_tokens': 4,
                    'ffn_layer': 'mlp',
                },
            },
            'small': {
                'factory': vit_small,
                'kwargs': {
                    'patch_size': 16,
                    'n_storage_tokens': 4,
                    'ffn_layer': 'mlp',
                },
            },
            'so400m': {
                'factory': vit_so400m,
                'kwargs': {
                    'patch_size': 16,
                    'n_storage_tokens': 4,
                    'ffn_layer': 'swiglu',
                },
            },
            'huge2': {
                'factory': vit_huge2,
                'kwargs': {
                    'patch_size': 16,
                    'n_storage_tokens': 4,
                    'ffn_layer': 'mlp',
                },
            },
            '7b': {
                'factory': vit_7b,
                'kwargs': {
                    'patch_size': 16,
                    'n_storage_tokens': 4,
                    'ffn_layer': 'swiglu',
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

        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict, strict=True)

        print('[INFO][Detector::loadModel]')
        print('\t model loaded from:', model_file_path)
        return True

    @torch.no_grad()
    def detect(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_dtype = image_tensor.dtype
        image_device = image_tensor.device

        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        dino_features_dict = self.model.forward_features(image_tensor)

        assert isinstance(dino_features_dict, dict)

        cls_token = dino_features_dict["x_norm_clstoken"].unsqueeze(1)
        storage_tokens = dino_features_dict["x_storage_tokens"]
        patch_tokens = dino_features_dict["x_norm_patchtokens"]
        x_norm = torch.cat([cls_token, storage_tokens, patch_tokens], dim=1)

        x_norm = x_norm.to(image_device, dtype=image_dtype)

        return x_norm

    @torch.no_grad()
    def detectFile(self, image_file_path: str) -> Union[torch.Tensor, None]:
        if not os.path.exists(image_file_path):
            print('[ERROR][Detector::detectFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        image = Image.open(image_file_path)

        image = image.convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0)

        dino_feature = self.detect(image_tensor)

        return dino_feature
