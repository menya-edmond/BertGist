"""
Module contains models necessary to perform epidemiological corpus tagging and returning their label as either relevant or irrelevant

"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.optim import AdamW

__author__ = "Edmond Menya"
__email__ = "edmondmenya@gmail.com"

class EpidBioELECTRA(nn.Module):
    """
    BertGist Model addopted from pretrained BERT on huggingface library built on pytorch classes
    """

    def __init__(self,device):
        """
        Constructor for the deep attention model, layers are adopted from pytorch and weights pretrained on BioELECTRA LM
        :param no_target_classes: no of corpus target classes defines models last layer architecture
        """
        super().__init__()
        self.dv = device
        self.bert = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=1)
        """self.dropout = nn.Dropout(p=0.1)
        self.batchnorm=nn.BatchNorm1d(1,affine=False)
        self.output = nn.Linear(self.bert.config.hidden_size, no_target_classes)
        self.output = nn.Linear(self.bert.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)"""

    def forward(self, input_ids):
        """
        Main network architecture as defined in pytorch library.
        :param input_ids: token position ids for input corpus words as tokenized by model pretrained tokenizer
        :param attention_mask: attention mask that corresponds with input_id token positions
        :return: probability distribution over predicted corpus classes, max value of the predicted probability distribution
        """
            
        output = self.bert(input_ids)

        
        return output
