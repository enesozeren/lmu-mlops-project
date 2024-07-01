# import lightning
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
from utils.utils_functions import get_datasets, preprocessing
import random


# Hyperparameters
random_seed = 76
batch_size = 32
epochs = 2

# Set random seed for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
# if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(random_seed)

# Get the training, validation, and test datasets
(train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels) = get_datasets()

# TODO: Delete section (Take the first 100 samples from training and validation datasets)
train_tweets, train_labels = train_tweets[:100], train_labels[:100]
val_tweets, val_labels = val_tweets[:100], val_labels[:100]


######################
### Pre-processing ###
######################
# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# TODO: Refactoring -> move function to utils
def tokenize_tweets(tweets):
    token_ids = []
    attention_masks = []
    for sample in tweets:
        encoding_dict = preprocessing(sample, tokenizer)
        token_ids.append(encoding_dict["input_ids"])
        attention_masks.append(encoding_dict["attention_mask"])
    token_ids = torch.cat(token_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return token_ids, attention_masks


train_token_ids, train_attention_masks = tokenize_tweets(train_tweets)
val_token_ids, val_attention_masks = tokenize_tweets(val_tweets)

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

train_set = TensorDataset(train_token_ids, train_attention_masks, train_labels)
val_set = TensorDataset(val_token_ids, val_attention_masks, val_labels)

train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
validation_dataloader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=batch_size)


################
#### Model #####
################
class HatespeechModel(LightningModule):
    def __init__(self):
        super(HatespeechModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def _b_tp(self, preds, labels):
        """Returns True Positives (TP): count of correct predictions of actual class 1"""
        return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

    def _b_fp(self, preds, labels):
        """Returns False Positives (FP): count of wrong predictions of actual class 1"""
        return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

    def _b_tn(self, preds, labels):
        """Returns True Negatives (TN): count of correct predictions of actual class 0"""
        return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

    def _b_fn(self, preds, labels):
        """Returns False Negatives (FN): count of wrong predictions of actual class 0"""
        return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

    def b_metrics(self, preds, labels):
        """
        Returns the following metrics:
            - accuracy    = (TP + TN) / N
            - precision   = TP / (TP + FP)
            - recall      = TP / (TP + FN)
            - specificity = TN / (TN + FP)
        """
        preds = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()
        tp = self._b_tp(preds, labels)
        # TODO: tp = sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])
        tn = self._b_tn(preds, labels)
        fp = self._b_fp(preds, labels)
        fn = self._b_fn(preds, labels)
        b_accuracy = (tp + tn) / len(labels)
        b_precision = tp / (tp + fp) if (tp + fp) > 0 else "nan"
        b_recall = tp / (tp + fn) if (tp + fn) > 0 else "nan"
        b_specificity = tn / (tn + fp) if (tn + fp) > 0 else "nan"
        return b_accuracy, b_precision, b_recall, b_specificity

    # TODO: Do wee need a forward pass? output vs outputs in training_step
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch):
        input_ids, attention_mask, labels = batch  # input_itds represents token IDs
        outputs = self(input_ids, attention_mask, labels=labels)  # model forward pass
        loss = outputs.loss  # TODO: Is outputs an object that contains loss as attribute?
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        # TODO: add train accuracy
        return loss

    # TODO: implement def on_train_epoch_end(self): ... ?

    def validation_step(self, batch):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels=labels)
        val_loss = outputs.loss
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = labels.to("cpu").numpy()
        b_accuracy, b_precision, b_recall, b_specificity = self.b_metrics(logits, label_ids)

        # TODO: check lightning standardisierte Metriken and revise code accodringly
        self.log("val_loss", val_loss)
        self.log("val_accuracy", b_accuracy)  # TODO: check if accuracy is for batch or for whole dataset/one epoch
        if b_precision != "nan":
            self.log("val_precision", b_precision)
        if b_recall != "nan":
            self.log("val_recall", b_recall)
        if b_specificity != "nan":
            self.log("val_specificity", b_specificity)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, eps=1e-08)
        return optimizer


model = HatespeechModel()

################
### Callback ###
################
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, mode="min"
)

################
### Training ###
################
wandb_logger = WandbLogger(project="hatespeech_detection", job_type="train")

# TODO: progress bar, gpu accelaration dynamically
trainer = Trainer(
    logger=wandb_logger,
    max_epochs=epochs,
    # gpus=1 if torch.cuda.is_available() else 0,
    devices=1,
    accelerator="gpu",
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_dataloader, validation_dataloader)
