import torch.nn as nn
from torchvision import datasets, models, transforms


class VGG16FT(nn.Module):
    def __init__(self, n_classes):
        super(VGG16FT, self).__init__()
        self.vgg = models.vgg16(pretrained=True)

    def forward(self, x):
        output = self.vgg(x)
        features = self.vgg.features(x) #self.vgg.classifier[0](intermedia_out)
        return features, output