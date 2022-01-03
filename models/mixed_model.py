import torch
import torch.nn as nn
from models.bert import BertForSentimentClassification
from models.vgg import VGG16FT
from models.classification import Classification
from models.vit import ViTFT

class MixModel(nn.Module):
    def __init__(self, config, n_classes=2, bert_path=None, vit = False, vgg_path=None, classification_path=None):
        super(MixModel, self).__init__()
        if bert_path is not None:
            self.bert = BertForSentimentClassification.from_pretrained(bert_path)
        else:
            self.bert = BertForSentimentClassification(config)

        text_feature_dim = config.hidden_size

        if vit:
            self.image_feature_extractor = ViTFT
            image_feature_dim = 4096 #TODO change this value
        else:
            self.image_feature_extractor = VGG16FT(n_classes)
            if vgg_path is not None:
                self.image_feature_extractor.load_state_dict(torch.load(vgg_path))

            image_feature_dim = self.image_feature_extractor.vgg.features[-3].out_channels * 6 * 6 # the 6 is calculated inside from pytorchconv2d

        linear_input_dimension = text_feature_dim + image_feature_dim #614400#config.hidden_size + 4096 # 4096 è il numero di features estrate da vgg

        if classification_path is not None:
            self.classification = Classification(linear_input_dimension, n_classes)
            self.classification.load_state_dict(torch.load(classification_path))
        else:
            self.classification = Classification(linear_input_dimension, n_classes)





    def forward(self, text, text_input_mask, image):

        with torch.no_grad():
            text_features, logit = self.bert.forward(text, text_input_mask)
            image_features, clas = self.vgg(image)

        mixed_features = torch.cat((torch.flatten(text_features, start_dim=1), torch.flatten(image_features, start_dim=1)), dim=1)
        classification = self.classification(mixed_features)
        return classification
