"""
CLIP utility functions for the DebiasDiffusion project.

This module provides a CLIPEncoder class that wraps the CLIP model
for encoding images and text, and computing similarities between them.
It's designed to be used for various tasks in the DebiasDiffusion project
that require image-text similarity computations.

Usage:
    from src.utils.clip_utils import CLIPEncoder

    encoder = CLIPEncoder()
    similarities = encoder(images, texts)
"""

import torch
import clip
from PIL import Image
import torch.nn as nn
from typing import List, Union

class CLIPEncoder(nn.Module):
    """
    A wrapper class for the CLIP model to encode images and text,
    and compute similarities between them.
    """

    def __init__(self):
        """Initialize the CLIPEncoder with the CLIP model."""
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.cos_similarity = nn.CosineSimilarity(dim=1)

    def encode_images_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode a batch of images using the CLIP model.

        Args:
            images (List[Image.Image]): A list of PIL Image objects.

        Returns:
            torch.Tensor: Encoded image features.
        """
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
        return image_features.float()

    def encode_texts_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts using the CLIP model.

        Args:
            texts (List[str]): A list of text strings.

        Returns:
            torch.Tensor: Encoded text features.
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.float()

    def compute_similarities(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarities between two sets of features.

        Args:
            features1 (torch.Tensor): First set of features.
            features2 (torch.Tensor): Second set of features.

        Returns:
            torch.Tensor: Cosine similarities between the feature sets.
        """
        return self.cos_similarity(features1, features2)

    def forward(self, images: List[Image.Image], texts_or_images: Union[List[str], List[Image.Image]]) -> torch.Tensor:
        """
        Compute similarities between images and texts or other images.

        Args:
            images (List[Image.Image]): A list of PIL Image objects.
            texts_or_images (Union[List[str], List[Image.Image]]): A list of text strings or PIL Image objects.

        Returns:
            torch.Tensor: Similarities between the input images and texts/images.
        """
        image_features = self.encode_images_batch(images)
        
        if isinstance(texts_or_images[0], str):
            # Text similarity
            text_features = self.encode_texts_batch(texts_or_images)
            similarities = self.compute_similarities(image_features, text_features)
        else:
            # Image similarity
            other_image_features = self.encode_images_batch(texts_or_images)
            similarities = self.compute_similarities(image_features, other_image_features)
        
        return similarities