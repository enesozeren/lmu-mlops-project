from pytorch_lightning import LightningModule
import torch
from transformers import BertForSequenceClassification


class BertClassifier(LightningModule):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch):
        return batch

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.model(input_ids, attention_mask)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)
