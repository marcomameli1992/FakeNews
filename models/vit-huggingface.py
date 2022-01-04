from transformers import ViTFeatureExtractor
from torch import nn

class ViTFT(nn.Module):

    def __init__(self):
        super(ViTFT, self).__init__()
        self.vit = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', size=128)
        #self.conv = nn.Conv2d(3, 64, 3, )

    def forward(self, x):
        features = self.vit(x, return_tensors='pt')['pixel_values']

        return features#self.conv(features)

if __name__ == "__main__":
    import torch
    from skimage.io import imread
    from torchvision import transforms
    feature_extractor = ViTFT()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    img_rand = imread('/Users/marcomameli01/Downloads/Dataset/test/1a1dfb.jpg')
    transform = transforms.ToTensor()
    img_rand = transform(img_rand)
    img_feature = feature_extractor(torch.unsqueeze(img_rand, dim=0))

