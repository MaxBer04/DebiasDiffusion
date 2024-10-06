"""
Classifier Models for DebiasDiffusion

This module provides classifier models used in the DebiasDiffusion project for
attribute prediction in the h-space of diffusion models. It includes implementations
of linear and ResNet18-based classifiers.

Classes:
    GenderClassifier: Linear classifier for gender prediction.
    ResNet18GenderClassifier: ResNet18-based classifier for gender prediction.

Functions:
    make_classifier_model: Factory function to create classifier models.

Usage:
    from src.utils.classifier import make_classifier_model

    classifier = make_classifier_model(
        in_channels=1280,
        image_size=8,
        out_channels=2,
        model_type="linear"
    )
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Literal

# Constants
TIMESTEPS: List[int] = [
    980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740,
    720, 700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460,
    440, 420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180,
    160, 140, 120, 100, 80, 60, 40, 20, 0
]

class GenderClassifier(nn.Module):
    """Linear classifier for gender prediction."""

    def __init__(self, in_channels: int, image_size: int, out_channels: int, prefix: str = None):
        """
        Initialize the GenderClassifier.

        Args:
            in_channels (int): Number of input channels.
            image_size (int): Size of the input image.
            out_channels (int): Number of output channels (classes).
            prefix (str, optional): Prefix for the model. Defaults to None.
        """
        super().__init__()
        self.input_dim = in_channels * image_size * image_size
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, out_channels) for _ in range(50)])
        self.prefix = prefix

    def forward(self, x: torch.Tensor, timesteps: List[int]) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            timesteps (List[int]): List of timesteps for each sample in the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        x = x.reshape(x.shape[0], -1)
        batch_size = x.shape[0]

        if isinstance(timesteps, int):
            timesteps = [timesteps] * batch_size

        selected_linears = torch.stack([self.linears[i].weight for i in timesteps]).to(dtype=x.dtype)
        selected_biases = torch.stack([self.linears[i].bias for i in timesteps]).to(dtype=x.dtype)

        output = torch.bmm(x.unsqueeze(1), selected_linears.transpose(1, 2)).squeeze(1) + selected_biases

        return output

class ResNet18GenderClassifier(nn.Module):
    """ResNet18-based classifier for gender prediction."""

    def __init__(self, in_channels: int, image_size: int, out_channels: int, prefix: str = None):
        """
        Initialize the ResNet18GenderClassifier.

        Args:
            in_channels (int): Number of input channels.
            image_size (int): Size of the input image.
            out_channels (int): Number of output channels (classes).
            prefix (str, optional): Prefix for the model. Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.out_channels = out_channels
        self.prefix = prefix

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.time_fcs = nn.ModuleList([nn.Linear(num_ftrs, out_channels) for _ in range(50)])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            t (torch.Tensor): Tensor of timesteps for each sample in the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        features = self.resnet(x)

        timestep_indices = torch.tensor([TIMESTEPS.index(ti.item()) for ti in t], device=x.device)
        batch_size = x.shape[0]

        selected_fcs = torch.stack([self.time_fcs[i].weight for i in timestep_indices])
        selected_biases = torch.stack([self.time_fcs[i].bias for i in timestep_indices])

        output = torch.bmm(features.unsqueeze(1), selected_fcs.transpose(1, 2)).squeeze(1) + selected_biases

        return output

def make_classifier_model(
    in_channels: int,
    image_size: int,
    out_channels: int,
    prefix: str = "train",
    model_type: Literal["linear", "resnet18"] = "linear"
) -> nn.Module:
    """
    Factory function to create classifier models.

    Args:
        in_channels (int): Number of input channels.
        image_size (int): Size of the input image.
        out_channels (int): Number of output channels (classes).
        prefix (str, optional): Prefix for the model. Defaults to "train".
        model_type (str, optional): Type of model to create. Choices are "linear" or "resnet18". Defaults to "linear".

    Returns:
        nn.Module: The created classifier model.

    Raises:
        ValueError: If an invalid model_type is provided.
    """
    if model_type == "linear":
        return GenderClassifier(in_channels, image_size, out_channels, prefix)
    elif model_type == "resnet18":
        return ResNet18GenderClassifier(in_channels, image_size, out_channels, prefix)
    else:
        raise ValueError("Invalid model_type. Choose 'linear' or 'resnet18'.")