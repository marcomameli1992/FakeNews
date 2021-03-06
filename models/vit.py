import torch
import timm
from torch import nn

class ViTFT(nn.Module):

    def __init__(self):
        super(ViTFT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        #self.conv = nn.Conv2d(3, 64, 3, )

    def forward(self, x):

        return self.vit.forward_features(x)


if __name__ == '__main__':
    import torch
    from skimage.io import imread
    from skimage.transform import resize
    from torchvision import transforms

    feature_extractor = ViTFT()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    img_rand = resize(imread('/home/vrai/FakeNews2/Dataset/test/1a1dfb.jpg'), (224, 224, 3))
    transform = transforms.ToTensor()
    img_rand = transform(img_rand)
    img_feature = feature_extractor(torch.unsqueeze(img_rand.type(torch.float).to(device), dim=0))