"""
Module contains methods to train and test epidemiological corpus taggers found in models.py

"""

import torch
import pickle
import json
import numpy as np
import torch.nn as nn
from models.classes import Models
from collections import defaultdict, OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.optim import AdamW

import evaluate

__author__ = "Edmond Menya"
__email__ = "edmondmenya@gmail.com"

def train_epoch(model,train_dataloader,optimizer,device,scheduler): #here
    """
    Trains model for every epoch
    :param model: object of food security model being run
    :param data_loader: object of train dataset
    :param optimizer: optimizer algorithm to be used in training
    :param device: GPU device being used to run model
    :param scheduler: scheduler value to reduce learning rate as training progresses
    :return: computed train accuracy, computed train loss
    """
    model = model.train()
    losses = []
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return np.mean(losses)


def model_eval(model,test_dataloader,device): #here
    """
    Evaluates the model after training by computing test accuracy and error rates
    :param model: object of epid model being tested
    :param data_loader: object of test dataset
    :param device: GPU device being used to run model
    :return: test accuracy,test loss,f_score value,precision value,recall value
    """
    metric = evaluate.load("mse")
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)
            

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=logits, references=batch["labels"])

    return metric.compute(squared=False)

def k_fold_cross_val(epochs,device,model_choice,tokenizer,train_dataloader, test_dataloader):
    """
    Trains and tests epid model using cross validation technigue and averages the cross validated models performance
    :param num_folds: value of k for the k-fold corss validator
    :param epochs: number of iterations to run for every k-fold instance
    :param batch_size: the set batch value for dataset segmentations
    :param max_len: the set max token length per corpus
    :param input_ids: token position ids for input corpus words as tokenized by model pretrained tokenizer
    :param labels: labels for corpus in dataset
    :param attention_masks: attention mask that corresponds with input_id token positions
    :param device: GPU device being used to run k-folded models
    :return: accuracy for each fold,f_score value for each fold,precision value for each fold,recall value for each fold
    """
    acc_per_fold = []


    if model_choice == 'bert':
        model = Models.EpidBioELECTRA()
        
    model.to(device)
    

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = epochs * len(train_dataloader)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    history = defaultdict(list)
    loss_values = []


    for epoch in range(epochs):
        
        total_loss = 0
        print(f'======== Epoch {epoch+1}/{epochs} ========')
        train_loss = train_epoch(model,train_dataloader,optimizer,device,scheduler) #here
        #train_acc,train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,scheduler,max_grad_norm)
        #train_acc = train_acc/normalizer
        #print(f'Train Loss: {train_loss} Train Accuracy: {train_acc}')
        total_loss += train_loss.item()

        avg_train_loss = total_loss / len(train_dataloader.dataset)  
        loss_values.append(avg_train_loss)

        #val_acc,val_loss,_,_,_,_,_ = model_eval(model,valid_dataloader,device) #here
        
        #print(f'Val Loss: {val_loss} Val Accuracy: {val_acc}')

        history['train_loss'].append(train_loss)
        #history['train_acc'].append(train_acc)

        #history['val_loss'].append(val_loss)
        #history['val_acc'].append(val_acc)

    test_rmse = model_eval(model,test_dataloader,device)
    #test_acc,test_loss,f_score,prec,rec = model_eval(model,test_dataloader,loss_fn)
    #test_acc = test_acc
    print(f'Test Loss is: {test_rmse}')
    acc_per_fold.append(test_rmse)
    """f_score_per_fold.append(f_score)
    prec_per_fold.append(prec)
    rec_per_fold.append(rec)
    org_labels_per_fold[fold_no] = original_labels
    pred_probs_per_fold[fold_no] = pred_probs"""




    return acc_per_fold