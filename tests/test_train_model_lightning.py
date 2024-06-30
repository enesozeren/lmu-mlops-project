from mlops_project.train_model_lightning import HatespeechModel
import torch
from transformers import BertTokenizer


def test_bert_classifier_training_step():
    # given
    model = HatespeechModel()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Create some dummy input data
    sample_text = ["This is a test tweet", "Another test tweet"]
    encoding = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = torch.tensor([1, 0])

    # Create a dummy batch
    batch = (input_ids, attention_mask, labels)

    # Perform a training step
    loss = model.training_step(batch)  # , 0

    # Check if the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.data == torch.Tensor(0.6979)
    assert loss.dim() == 0
