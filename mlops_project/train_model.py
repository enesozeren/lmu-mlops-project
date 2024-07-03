from utils.utils_functions import get_datasets, tokenize_tweets
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_lightning import Trainer
from hate_speech_model import HatespeechModel

# Hyperparameters
# RANDOM_SEED = 76
BATCH_SIZE = 32
EPOCHS = 2

# Get the training, validation, and test datasets
(train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels) = get_datasets()

train_token_ids, train_attention_masks = tokenize_tweets(train_tweets)
val_token_ids, val_attention_masks = tokenize_tweets(val_tweets)

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

train_set = TensorDataset(train_token_ids, train_attention_masks, train_labels)
val_set = TensorDataset(val_token_ids, val_attention_masks, val_labels)

train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=BATCH_SIZE)


# Training
model = HatespeechModel()
trainer = Trainer(
    # logger=wandb_logger,
    max_epochs=EPOCHS,
    # devices=1,
    # accelerator="gpu",
    # callbacks=[checkpoint_callback],
)
trainer.fit(model, train_dataloader, validation_dataloader)
