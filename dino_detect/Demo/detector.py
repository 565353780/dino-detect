import os
import cv2
import torch
import numpy as np

from tqdm import trange

from dino_detect.Module.detector import Detector


def demo():
    model_type = "huge+"
    model_file_path = (
        os.environ["HOME"] + "/chLi/Model/DINOv3/dinov3_vitl16_pretrain.pth"
    )
    model_file_path = None
    dtype = "auto"
    device = "cuda:0"

    image_file_path = "/home/chli/chLi2/Dataset/CapturedImage/ShapeNet/02691156/10155655850468db78d106ce0a280f87/y_5_x_3.png"
    if not os.path.exists(image_file_path):
        H, W = 512, 512  # 可以根据实际需要调整大小
        noise_img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
        cv2.imwrite(image_file_path, noise_img)

    detector = Detector(model_type, model_file_path, dtype, device)

    for _ in trange(100):
        dino_feature = detector.detect(
            torch.randn([3, 3, 518, 518], dtype=torch.float32, device="cpu")
        )

    print("dino_feature:")
    print(dino_feature)
    print(dino_feature.shape)

    for _ in trange(100):
        dino_feature = detector.detectFile(image_file_path)

    print("dino_feature:")
    print(dino_feature)
    print(dino_feature.shape)
    return True
