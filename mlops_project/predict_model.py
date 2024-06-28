import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer
from utils.utils_functions import read_dataset
from datetime import datetime

def predict(model_path: str, dataset_path: str) -> None:
    """Run prediction for a given model and dataset.
    Args:
        model_path: model to use for prediction
        dataset_path: dataset to use for prediction
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2, 
        output_attentions=False, 
        output_hidden_states=False
    )

    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Read the dataset
    tweets, _ = read_dataset(dataset_path)

    # Tokenize the dataset
    inputs = tokenizer(tweets, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Move inputs to the device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(predictions, dim=1)

    return probabilities

if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    args = parser.parse_args()

    # Run prediction
    predictions = predict(args.model_path, args.dataset_path)

    # Save predictions
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    PREDICTION_PATH = f'outputs/predictions/predictions_{current_time}.pt'
    torch.save(predictions, PREDICTION_PATH)
