# import os
import logging
from data.utils import get_datasets, preprocessing  # , b_metrics
from transformers import BertTokenizer  # BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# import random
# import numpy as np
# from tqdm import tqdm

# Suppress warnings from the transformers library
logging.basicConfig(level=logging.ERROR)

# Hyperparameters
random_seed = 76
batch_size = 32
epochs = 2


# Get the training, validation, and test datasets
(train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels) = get_datasets()

# delete
# Take the first 100 samples from training and validation datasets
train_tweets, train_labels = train_tweets[:100], train_labels[:100]
val_tweets, val_labels = val_tweets[:100], val_labels[:100]

######################
### Pre-processing ###
######################
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_token_ids = []
train_attention_masks = []
val_token_ids = []
val_attention_masks = []

for sample in train_tweets:
    encoding_dict = preprocessing(sample, tokenizer)
    train_token_ids.append(encoding_dict["input_ids"])
    train_attention_masks.append(encoding_dict["attention_mask"])

for sample in val_tweets:
    encoding_dict = preprocessing(sample, tokenizer)
    val_token_ids.append(encoding_dict["input_ids"])
    val_attention_masks.append(encoding_dict["attention_mask"])

train_token_ids = torch.cat(train_token_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_labels)

val_token_ids = torch.cat(val_token_ids, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
val_labels = torch.tensor(val_labels)

train_set = TensorDataset(train_token_ids, train_attention_masks, train_labels)
val_set = TensorDataset(val_token_ids, val_attention_masks, val_labels)

train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)

validation_dataloader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=batch_size)
