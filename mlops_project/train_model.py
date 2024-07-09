import torch
import argparse
import yaml
import wandb
import random
import numpy
from utils.utils_functions import get_datasets, tokenize_tweets
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_lightning.callbacks import ModelCheckpoint  # , EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from hate_speech_model import HatespeechModel


# Get the training, validation, and test datasets
(train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels) = get_datasets()

train_token_ids, train_attention_masks = tokenize_tweets(train_tweets)
val_token_ids, val_attention_masks = tokenize_tweets(val_tweets)

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

train_set = TensorDataset(train_token_ids, train_attention_masks, train_labels)
val_set = TensorDataset(val_token_ids, val_attention_masks, val_labels)


# Reproducibility
seed_everything(47, workers=True)
torch.manual_seed(47)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Script to run with a config file.")
parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file.")
args = parser.parse_args()

# Load YAML configuration file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Hyperparameter sweep
sweep_id = wandb.sweep(sweep=config, project="hate_speech_detection")


# Model initiliazation and training
def main():
    wandb.init(project="hate_speech_detection")
    train_dataloader = DataLoader(
        train_set,
        worker_init_fn=seed_worker,
        generator=g,
        sampler=RandomSampler(train_set),
        batch_size=wandb.config.BATCH_SIZE,
    )
    validation_dataloader = DataLoader(
        val_set,
        worker_init_fn=seed_worker,
        generator=g,
        sampler=SequentialSampler(val_set),
        batch_size=wandb.config.BATCH_SIZE,
    )

    # Initialize the model
    model = HatespeechModel(wandb.config.LEARNING_RATE)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="mlops_project/checkpoints", filename="best-checkpoint", save_top_k=1, mode="min"
    )
    # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # computations in 16-bit to speed up training, model weights in 32-bit to maintain accuracy
        deterministic=True,
        max_epochs=wandb.config.EPOCHS,
        limit_train_batches=0.04,
        limit_val_batches=0.04,
        logger=WandbLogger(project="hate_speech_detection"),
        callbacks=[checkpoint_callback],  # , early_stopping_callback
    )

    # Train the model
    trainer.fit(model, train_dataloader, validation_dataloader)


wandb.agent(sweep_id, function=main, count=3)
