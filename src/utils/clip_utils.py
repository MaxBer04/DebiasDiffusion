import torch
import clip
from PIL import Image
import torch.nn as nn

class CLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.cos_similarity = nn.CosineSimilarity(dim=1)

    def encode_images_batch(self, images):
        # Expects PIL Image objects
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
        return image_features.float()

    def encode_texts_batch(self, texts):
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.float()

    def compute_similarities(self, features1, features2):
        return self.cos_similarity(features1, features2)

    def forward(self, images, texts_or_images):
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