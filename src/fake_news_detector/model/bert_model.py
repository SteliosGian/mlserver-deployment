"""
Bert model
"""
import torch.nn as nn
from transformers import BertForSequenceClassification


class BERT(nn.Module):
    def __init__(self, pretrained_model: str = "bert-base-uncased", num_labels: int = 2):
        super(BERT, self).__init__()

        model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                              num_labels=num_labels,
                                                              output_attentions=False,
                                                              output_hidden_states=False,
                                                              )
        return model
