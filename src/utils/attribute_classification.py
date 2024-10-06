"""
Attribute classification utilities for the DebiasDiffusion project.

This module provides functionality for classifying attributes (gender, race, age)
in images using pre-trained models. It includes a custom AttributeClassifier class
and utility functions for loading and using these classifiers.

Usage:
    from src.utils.attribute_classification import get_attribute_classifier, classify_attribute

    gender_classifier = get_attribute_classifier('gender', device)
    gender_probs = classify_attribute(face_image, gender_classifier, 'gender')
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Union

import torch
import torchvision
from torchvision import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

class AttributeClassifier:
    """A class for classifying attributes in images."""

    def __init__(self, attribute_type: str, device: torch.device, use_race7: bool = False, use_legacy_gender: bool = False):
        """
        Initialize the AttributeClassifier.

        Args:
            attribute_type (str): The type of attribute to classify ('gender', 'race', or 'age').
            device (torch.device): The device to run the model on.
            use_race7 (bool, optional): Whether to use 7-class race classification. Defaults to False.
            use_legacy_gender (bool, optional): Whether to use the legacy gender classifier. Defaults to False.
        """
        self.attribute_type = attribute_type
        self.device = device
        self.use_race7 = use_race7
        self.use_legacy_gender = use_legacy_gender
        self.model = self.load_model()
        
    def load_model(self) -> torch.nn.Module:
        """
        Load the appropriate classification model based on the attribute type.

        Returns:
            torch.nn.Module: The loaded classification model.
        """
        if self.attribute_type == 'gender' and self.use_legacy_gender:
            model = torchvision.models.resnet34(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, 18)
            weight_path = BASE_DIR / "data" / "model_data" / "external_classifiers" / "from_fairface" / "res34_fair_align_multi_7_20190809.pt"
        else:
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            if self.attribute_type == 'gender':
                model._modules['classifier'][3] = torch.nn.Linear(1280, 2, bias=True)
                weight_path = BASE_DIR / "data" / "model_data" / "external_classifiers" / "from_FDM" / "CelebA-MobileNetLarge-Gender-09191318" / "epoch=19-step=25320_MobileNetLarge.pt"
            elif self.attribute_type == 'race':
                model._modules['classifier'][3] = torch.nn.Linear(1280, 4, bias=True)
                weight_path = BASE_DIR / "data" / "model_data" / "external_classifiers" / "from_FDM" / "fairface-MobileNetLarge-Race4-09191318" / "epoch=19-step=6760_MobileNetLarge.pt"
            elif self.attribute_type == 'age':
                model._modules['classifier'][3] = torch.nn.Linear(1280, 2, bias=True)
                weight_path = BASE_DIR / "data" / "model_data" / "external_classifiers" / "from_FDM" / "fairface-MobileNetLarge-Age2-09191319" / "epoch=19-step=6760_MobileNetLarge.pt"
            else:
                raise ValueError(f"Unsupported attribute type: {self.attribute_type}")
        
        if not weight_path.exists():
            raise FileNotFoundError(f"The weight file does not exist at: {weight_path}")
        
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def classify(self, face_chip: Union[torch.Tensor, Any]) -> torch.Tensor:
        """
        Classify the attribute in the given face image.

        Args:
            face_chip (Union[torch.Tensor, Any]): The face image to classify.

        Returns:
            torch.Tensor: The classification probabilities.
        """
        if isinstance(face_chip, torch.Tensor):
            if face_chip.dim() == 3:
                face_chip = face_chip.unsqueeze(0)
            face_image = face_chip.float()
            if face_image.max() > 1:
                face_image = face_image / 255.0
        else:
            to_tensor = transforms.ToTensor()
            face_image = to_tensor(face_chip).unsqueeze(0)
        
        face_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face_image)
        face_image = face_image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(face_image)
        
        if self.attribute_type == 'gender' and self.use_legacy_gender:
            outputs = outputs.cpu().numpy().squeeze()
            gender_probs = torch.softmax(torch.tensor(outputs[7:9]), dim=0)
            return gender_probs
        else:
            probs = torch.softmax(outputs, dim=1).cpu().squeeze()
            if self.attribute_type == 'gender':
                return probs.flip(0)  # Reverse for [male, female]
            elif self.attribute_type == 'race' and not self.use_race7:
                return probs[[0, 1, 3, 2]]  # Swap Asian and Indian
            else:
                return probs

def get_attribute_classifier(attribute_type: str, device: torch.device, use_race7: bool = False, use_legacy_gender: bool = False) -> AttributeClassifier:
    """
    Get an attribute classifier for the specified attribute type.

    Args:
        attribute_type (str): The type of attribute to classify ('gender', 'race', or 'age').
        device (torch.device): The device to run the model on.
        use_race7 (bool, optional): Whether to use 7-class race classification. Defaults to False.
        use_legacy_gender (bool, optional): Whether to use the legacy gender classifier. Defaults to False.

    Returns:
        AttributeClassifier: An instance of the AttributeClassifier for the specified attribute.
    """
    return AttributeClassifier(attribute_type, device, use_race7, use_legacy_gender)

def classify_attribute(face_chip: Union[torch.Tensor, Any], attribute_classifier: AttributeClassifier, attribute_type: str) -> torch.Tensor:
    """
    Classify the specified attribute in the given face image.

    Args:
        face_chip (Union[torch.Tensor, Any]): The face image to classify.
        attribute_classifier (AttributeClassifier): The attribute classifier to use.
        attribute_type (str): The type of attribute to classify ('gender', 'race', or 'age').

    Returns:
        torch.Tensor: The classification probabilities.
    """
    return attribute_classifier.classify(face_chip)