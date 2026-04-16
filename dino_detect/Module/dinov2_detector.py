import os
import torch

from torchvision import transforms
from transformers import AutoModel

from dino_detect.Method.detect import (
    preprocessImages,
    postprocessFeatures,
    detectFile,
    toPCAImages,
)


class DINOv2Detector:
    """Drop-in replacement for dino_detect.Module.detector.Detector.

    Uses a HuggingFace DINOv2 model instead of the custom DINOv3 ViT.
    Exposes the same ``__init__`` parameter names (``model_file_path`` maps to
    the local HuggingFace repo directory), the ``is_valid`` flag, and the
    ``detect(image_tensor)`` method so that callers (Trainer / Detector) can
    swap between the two without changing any surrounding code.
    """

    def __init__(
        self,
        model_file_path: str,
        model_type: str = "dinov2-large",
        dtype="auto",
        device: str = "cpu",
    ) -> None:
        self.device = device
        if dtype == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = dtype

        self.is_valid = False

        repo_dir = os.path.realpath(model_file_path)
        if not os.path.isdir(repo_dir):
            print("[ERROR][DINOv2Detector::__init__]")
            print("\t model directory not found:", repo_dir)
            return

        self.model = AutoModel.from_pretrained(
            repo_dir,
            local_files_only=True,
        ).to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.is_valid = True

    @torch.no_grad()
    def detect(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Match the DINODetector.detect() contract.

        Args:
            image_tensor: [B, H, W, 3], float32, range [0, 1]

        Returns:
            x_norm: [B, N, C] where the first 5 tokens are
                    [cls, pad, pad, pad, pad] so that ``x_norm[:, 5:]``
                    yields pure patch tokens (same convention as DINODetector).
        """
        image_tensor, input_dtype, input_device = preprocessImages(
            image_tensor, self.transform, self.device, self.dtype,
        )

        dino_out = self.model(pixel_values=image_tensor)
        feats = dino_out.last_hidden_state

        B, T, C = feats.shape
        cls_token = feats[:, :1, :]
        patch_tokens = feats[:, 1:, :]

        storage_pad = torch.zeros(B, 4, C, device=feats.device, dtype=feats.dtype)
        x_norm = torch.cat([cls_token, storage_pad, patch_tokens], dim=1)

        return postprocessFeatures(x_norm, input_device, input_dtype)

    @torch.no_grad()
    def detectFile(self, image_file_path: str):
        return detectFile(self, image_file_path)

    @staticmethod
    @torch.no_grad()
    def toPCAImages(dino_feats: torch.Tensor):
        return toPCAImages(dino_feats)
