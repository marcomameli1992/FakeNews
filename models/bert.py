import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # The classification layer that takes the [CLS] representation and outputs the logit
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        '''
		Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)
		'''
        # Feed the input to Bert model to obtain outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        cls_reps = outputs.last_hidden_state[:, 0]
        # cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return cls_reps, logits

if __name__ == "__main__":
    import torch
    from transformers import AutoConfig, AutoTokenizer

    bertModelNameOrPath = 'bert-base-uncased'
    config = AutoConfig.from_pretrained(bertModelNameOrPath)
    feature_extractor = BertForSentimentClassification(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
