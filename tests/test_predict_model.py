import pytest
import torch
import os
from mlops_project.predict_model import predict
from transformers import BertForSequenceClassification

# Setup and Teardown using pytest fixtures
@pytest.fixture
def setup_files():
    # Create temporary model and dataset files
    model_path = 'temp_model.pt'
    dataset_path = 'temp_dataset.txt'
    # Load the model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2, 
        output_attentions=False, 
        output_hidden_states=False
    )
    torch.save(model.state_dict(), model_path)
    
    with open(dataset_path, 'w') as f:
        f.write("Sample text\n" * 10)  # Dummy dataset with 10 samples
    yield model_path, dataset_path
    # Cleanup
    os.remove(model_path)
    os.remove(dataset_path)

# Test output of the predict function
def test_predict_output_type(setup_files):
    model_path, dataset_path = setup_files
    predictions = predict(model_path, dataset_path)
    assert isinstance(predictions, torch.Tensor), "Predictions should be a torch.Tensor"

# Test output shape of the predict function
def test_predict_output_shape(setup_files):
    model_path, dataset_path = setup_files
    predictions = predict(model_path, dataset_path)
    assert predictions.shape == (10, 2), "Output shape should be [N, 2] where N is the number of samples"

# Test output values of the predict function
def test_predict_output_values(setup_files):
    model_path, dataset_path = setup_files
    predictions = predict(model_path, dataset_path)
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1), "Probabilities should be between 0 and 1"
    assert torch.allclose(probabilities.sum(dim=1), torch.tensor([1.0] * 10)), "Probabilities should sum to 1"

# Test exception handling
def test_exception_handling():
    with pytest.raises(Exception):
        predict('non_existent_model.pt', 'non_existent_dataset.txt')
