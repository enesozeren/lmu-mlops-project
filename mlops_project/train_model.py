from utils.utils_functions import get_datasets, tokenize_tweets
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_lightning.callbacks import ModelCheckpoint  # , EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from hate_speech_model import HatespeechModel

# Hyperparameters
BATCH_SIZE = 4 * 32
EPOCHS = 20

# Reproducibility
seed_everything(47, workers=True)

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


# Train the model
model = HatespeechModel()

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", dirpath="mlops_project/checkpoints", filename="best-checkpoint", save_top_k=1, mode="min"
)

# early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

trainer = Trainer(
    accelerator="auto",
    devices="auto",
    precision="16-mixed",  # computations in 16-bit to speed up training, model weights in 32-bit to maintain accuracy
    deterministic=True,
    max_epochs=EPOCHS,
    # limit_train_batches=0.04,
    # limit_val_batches=0.04,
    logger=WandbLogger(project="hate_speech_detection"),
    callbacks=[checkpoint_callback],  # , early_stopping_callback
)
trainer.fit(model, train_dataloader, validation_dataloader)
