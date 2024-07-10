import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy, Precision, Recall, Specificity
from transformers import BertForSequenceClassification


class HatespeechModel(LightningModule):
    def __init__(self, lr):
        super(HatespeechModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_prec = Precision(task="binary")
        self.val_rec = Recall(task="binary")
        self.val_spec = Specificity(task="binary")
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        return logits

    def training_step(self, batch):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        # Training accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.criterion(logits, labels)
        self.log("val_loss", val_loss)

        # Validation metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_prec(preds, labels)
        self.log("val_prec", self.val_prec, on_step=False, on_epoch=True)
        self.val_rec(preds, labels)
        self.log("val_rec", self.val_rec, on_step=False, on_epoch=True)
        self.val_spec(preds, labels)
        self.log("val_spec", self.val_spec, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-08)
        return optimizer
