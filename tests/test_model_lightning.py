from model_lightning import BertClassifier


def test_model_lightning():
    model = BertClassifier()
    batch = "test"
    assert model.training_step(batch) == batch
