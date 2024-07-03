from pytorch_lightning import LightningModule
import torch
from transformers import BertForSequenceClassification


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

    # TODO: output vs outputs in training_step
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        return logits

    def training_step(self, batch):
        input_ids, attention_mask, labels = batch  # input_itds represents token IDs
        logits = self(input_ids, attention_mask)  # model forward pass
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)  # , on_epoch=True, on_step=False, prog_bar=True
        # TODO: add train accuracy
        return loss

    def validation_step(self, batch):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.criterion(logits, labels)
        self.log("val_loss", val_loss)
        # self.log("val_accuracy")
        # TODO: check if accuracy is for batch or for whole dataset/one epoch
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, eps=1e-08)
        return optimizer
