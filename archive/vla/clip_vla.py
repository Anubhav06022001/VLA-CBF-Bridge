import torch
import clip
import numpy as np
from vla.base import VLAPolicy

class CLIPVLA(VLAPolicy):
    def __init__(self, device= "cuda"):
        self.device = device
        self.model , self.preprocess = clip.load("ViT-B/32", device = device)
        self.model.eval()

    def encode_image(self,image):
        # image: H x W x 3, uint8
        image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
        image = image.float() / 255.0
        image = torch.nn.functional.interpolate(image, size=224)
        image = image.to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(image)
        return emb / emb.norm(dim = -1, keepdim= True)

    def encode_text(self, text):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        return emb/ emb.norm(dim = -1, keepdim= True)

    def act(self,object_images, language):
        """
        object_images: list of cropped RGB images
        """
        text_emb = self.encode_text(language)

        scores = []
        for img in object_images:
            img_emb = self.encode_image(img)
            score = (img_emb @ text_emb.T).item()
            scores.append(score)

        for i, score in enumerate(scores):
            print(f"Image {i}: {score:.4f}")
            
        return int(np.argmax(scores))
