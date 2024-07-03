from mlops_project.hate_speech_model import HatespeechModel
from transformers import BertTokenizer
import torch


def test_training_step():
    # given
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Create some dummy input data
    sample_text = ["She is such a slut.", "Another test tweet"]
    encoding = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = torch.tensor([1, 0])

    # Create a dummy batch
    batch = (input_ids, attention_mask, labels)
    # when
    model = HatespeechModel()
    loss = model.training_step(batch)
    # then
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    # https://pytorch.org/docs/stable/testing.html
    # torch.testing.assert_close(loss.data, torch.tensor(0.6543))