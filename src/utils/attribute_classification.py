import torch
import os
import torchvision
from torchvision import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
import numpy as np

class AttributeClassifier:
    def __init__(self, attribute_type, device, use_race7=False, use_legacy_gender=False):
        self.attribute_type = attribute_type
        self.device = device
        self.use_race7 = use_race7
        self.use_legacy_gender = use_legacy_gender
        self.model = self.load_model()
        
    def load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if self.attribute_type == 'gender' and self.use_legacy_gender:
            model = torchvision.models.resnet34(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, 18)
            weight_path = os.path.join(current_dir, "data", "legacy-fairface", "res34_fair_align_multi_7_20190809.pt")
        else:
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            if self.attribute_type == 'gender':
                model._modules['classifier'][3] = torch.nn.Linear(1280, 2, bias=True)
                weight_path = os.path.join(current_dir, "data", "5-trained-test-classifiers", "CelebA-MobileNetLarge-Gender-09191318", "epoch=19-step=25320_MobileNetLarge.pt")
            elif self.attribute_type == 'race':
                model._modules['classifier'][3] = torch.nn.Linear(1280, 4, bias=True)
                weight_path = os.path.join(current_dir, "data", "5-trained-test-classifiers", "fairface-MobileNetLarge-Race4-09191318", "epoch=19-step=6760_MobileNetLarge.pt")
            elif self.attribute_type == 'age':
                model._modules['classifier'][3] = torch.nn.Linear(1280, 2, bias=True)
                weight_path = os.path.join(current_dir, "data", "5-trained-test-classifiers", "fairface-MobileNetLarge-Age2-09191319", "epoch=19-step=6760_MobileNetLarge.pt")
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"The weight file does not exist at: {weight_path}")
        
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def classify(self, face_chip):
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
            gender_probs = np.exp(outputs[7:9]) / np.sum(np.exp(outputs[7:9]))
            return gender_probs  # Bereits in der Reihenfolge [male, female]
        else:
            probs = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
            if self.attribute_type == 'gender':
                return probs[::-1]  # Umkehren für [male, female]
            elif self.attribute_type == 'race' and not self.use_race7:
                return probs[[0, 1, 3, 2]]  # Vertauschen von Asian und Indian
            else:
                return probs

def get_attribute_classifier(attribute_type, device, use_race7=False, use_legacy_gender=False):
    return AttributeClassifier(attribute_type, device, use_race7, use_legacy_gender)

def classify_attribute(face_chip, attribute_classifier, attribute_type):
    return attribute_classifier.classify(face_chip)