import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import piq

class LPIPSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = piq.LPIPS().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        return self.preprocess(image).unsqueeze(0)

    def forward(self, images1, images2):
        batch_size = len(images1)
        lpips_values = []

        for i in range(batch_size):
            img1 = self.preprocess_image(images1[i]).to(self.device)
            img2 = self.preprocess_image(images2[i]).to(self.device)
            lpips_value = 1 - self.lpips(img1, img2)
            lpips_values.append(lpips_value)

        return torch.stack(lpips_values).squeeze()