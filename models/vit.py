from transformers import ViTFeatureExtractor
from torch import nn

class ViTFT(nn.Module):

    def __init__(self):
        super(ViTFT, self).__init__()
        self.vit = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, x):
        return self.vit(x)

if __name__ == "__main__":
    import torch
    feature_extractor = ViTFT()
    img_rand = torch.rand(3, 224, 224)
    img_feature = feature_extractor(img_rand)

