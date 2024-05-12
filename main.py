
import os
import re
import csv
import itertools
import json
import sys
import torch
import numpy as np
import sacremoses as ms
from optparse import OptionParser
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, AdamW, AutoTokenizer, AutoModelForMaskedLM, AutoModel, ElectraForPreTraining, ElectraTokenizerFast

from datasets import Dataset, load_dataset


if __name__ == "__main__":
    from data import dataGenerator
    import config as cfg
    from models import modelTrainTest as mtt
    from data.profile import Profile
    
    if not sys.argv:
        modelChoice=sys.argv[0]
        
    else:
        optparser = OptionParser()
        optparser.add_option('-a', '--modelChoice',
                             dest='modelChoice',
                             help='select model',
                             default=cfg.MODEL_CHOICE,
                             type='string')
        (options, args) = optparser.parse_args()
    
    modelChoice=options.modelChoice
    rel_data=cfg.rel_url
    irr_data=cfg.irr_url
    max_len=cfg.MAX_LEN
    batch_size=cfg.BATCH_SIZE
    num_folds=cfg.NUM_FOLDS
    epochs=cfg.EPOCHS
    #corpus_type=cfg.CORPUS_TYPE

    if modelChoice == 'bert':
        FOUNDATIONAL_LLM = "google-bert/bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(FOUNDATIONAL_LLM)



    # tell Pytorch to use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    if torch.cuda.device_count():
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    print('We are running ', modelChoice)
    
    #data = dataGenerator.data_generator(rel_data,irr_data,corpus_type)
    dataset = load_dataset("json", data_files={'train': 'data/train.json', 'test': 'data/test.json', 'validation':'data/val.json'})
    print(dataset['train'][10])

    #tokenized_datasets = dataGenerator.preprocess_function(dataset)
    def preprocess_function(examples):
        label = examples["score"] 
        examples = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        examples["labels"] = [float(s) for s in label]
        return examples

    tokenized_datasets = dataset.map(preprocess_function, remove_columns=["id", "uuid", "text", "score"], batched=True)
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8)
    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8)

    

    acc_per_fold = mtt.k_fold_cross_val(epochs,device,modelChoice,tokenizer,train_dataloader, test_dataloader)
            
    acc_list=[]
    for i in acc_per_fold:
        acc_list.append(np.round(i.tolist(),4))
        
    print(acc_list)




