"""
LPIPS Utility Module for DebiasDiffusion

This module provides a PyTorch implementation of the Learned Perceptual Image
Patch Similarity (LPIPS) metric. It uses a pre-trained VGG network to compute
perceptual similarities between images.

Usage:
    from src.utils.lpips_utils import LPIPSEncoder

    lpips_encoder = LPIPSEncoder()
    similarity = lpips_encoder(image1, image2)

Note:
    This module requires the PyTorch, torchvision, and PIQ libraries to be installed.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import piq
from typing import List, Union

class LPIPSEncoder(nn.Module):
    def __init__(self):
        """
        Initialize the LPIPSEncoder.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = piq.LPIPS().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess the input image.

        Args:
            image (Union[Image.Image, torch.Tensor]): Input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if isinstance(image, Image.Image):
            return self.preprocess(image).unsqueeze(0)
        return image.unsqueeze(0) if image.dim() == 3 else image

    def forward(self, images1: List[Union[Image.Image, torch.Tensor]], 
                images2: List[Union[Image.Image, torch.Tensor]]) -> torch.Tensor:
        """
        Compute LPIPS similarity between two lists of images.

        Args:
            images1 (List[Union[Image.Image, torch.Tensor]]): First list of images.
            images2 (List[Union[Image.Image, torch.Tensor]]): Second list of images.

        Returns:
            torch.Tensor: LPIPS similarity scores.
        """
        batch_size = len(images1)
        lpips_values = []

        for i in range(batch_size):
            img