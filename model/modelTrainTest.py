"""
Module contains methods to train and test epidemiological corpus taggers found in models.py

"""

import torch
import shap
import pickle
import json
import numpy as np
import torch.nn as nn
from models.classes import Models
from collections import defaultdict, OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,precision_score,recall_score
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.optim import AdamW

import evaluate

__author__ = "Edmond Menya"
__email__ = "edmondmenya@gmail.com"

def train_epoch(model,train_dataloader,       data_loader,loss_fn,optimizer,device,scheduler,max_grad_norm): #here
    """
    Trains model for every epoch
    :param model: object of epid model being run
    :param data_loader: object of train dataset
    :param loss_fn: loss function to be used in training
    :param optimizer: optimizer algorithm to be used in training
    :param device: GPU device being used to run model
    :param scheduler: scheduler value to reduce learning rate as training progresses
    :param max_grad_norm: grad norm value for gradient clipping
    :return: computed train accuracy, computed train loss
    """
    model = model.train()
    losses = []

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #for epoch in range(num_epochs):
    for batch in train_dataloader:
        #print(batch.items())
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return np.mean(losses)


def model_eval(model,test_dataloader,loss_fn,device): #here
    """
    Evaluates the model after training by computing test accuracy and error rates
    :param model: object of epid model being tested
    :param data_loader: object of test dataset
    :param loss_fn: loss function to be used in testing
    :param device: GPU device being used to run model
    :return: test accuracy,test loss,f_score value,precision value,recall value
    """
    metric = evaluate.load("mse")
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=logits, references=batch["labels"])

    return metric.compute(squared=False)

def k_fold_cross_val(num_folds,epochs,batch_size,max_len,input_ids,labels,attention_masks,device,model_choice,tokenizer,class_names):
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
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    split_size=.2
    acc_per_fold, f_score_per_fold, prec_per_fold, rec_per_fold = [],[],[],[]
    org_labels_per_fold, pred_probs_per_fold = {}, {}
    
    for train, test in kfold.split(input_ids, labels):
        print(f"\n ****************************** This is Fold Number {fold_no} ***************************** \n")
        
        no_target_classes=len(np.unique(labels))+1

        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids[train], labels[train],
                                                                random_state=2018, test_size=split_size)
        tr_masks, val_masks, _, _ = train_test_split(np.array(attention_masks)[train], input_ids[train],
                                                  random_state=2018, test_size=split_size)

        n_attention_masks = [[float(i != 0.0) for i in ii] for ii in tr_inputs]

        tr_masks, test_masks, _, _ = train_test_split(n_attention_masks, tr_inputs,
                                                  random_state=2018, test_size=split_size)

        tr_inputs, test_inputs, tr_tags, test_tags = train_test_split(tr_inputs, tr_tags,
                                                                  random_state=2018, test_size=split_size)

        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        test_inputs = torch.tensor(test_inputs)


        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        test_tags = torch.tensor(test_tags)

        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)
        test_masks = torch.tensor(test_masks)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size, drop_last=True)

        test_data = TensorDataset(test_inputs, test_masks, test_tags)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, drop_last=True)
        
        
        if model_choice == 'bert':
            model = Models.EpidBioELECTRA(device)
            
        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
              {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
              'weight_decay_rate': 0.01},
              {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
              'weight_decay_rate': 0.0}
          ]

        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_training_steps = epochs * len(train_dataloader)
        scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        
        history = defaultdict(list)
        best_accuracy = 0
        normalizer = batch_size*max_len
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

        test_rmse = model_eval(model,test_dataloader,loss_fn,device)
        #test_acc,test_loss,f_score,prec,rec = model_eval(model,test_dataloader,loss_fn)
        #test_acc = test_acc
        print(f'Test Loss for fold {fold_no}: {test_rmse} Test Accuracy: {test_acc} F Score: {f_score} Precision: {prec} Recall: {rec}')
        acc_per_fold.append(test_rmse)
        """f_score_per_fold.append(f_score)
        prec_per_fold.append(prec)
        rec_per_fold.append(rec)
        org_labels_per_fold[fold_no] = original_labels
        pred_probs_per_fold[fold_no] = pred_probs"""

        fold_no = fold_no + 1

        
      
    return acc_per_fold