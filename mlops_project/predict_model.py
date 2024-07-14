import argparse
import os
from datetime import datetime

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from utils.utils_functions import read_dataset


def predict(model_path: str, dataset_path: str) -> None:
    """Run prediction for a given model and dataset.
    Args:
        model_path: model to use for prediction
        dataset_path: dataset to use for prediction
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        state_dict=torch.load(model_path, map_location=device),
    )
    model.eval()

    # Read the dataset
    tweets, _ = read_dataset(dataset_path)

    # Tokenize the dataset
    inputs = tokenizer(tweets, return_tensors="pt", padding=True, truncation=True, max_length=512)

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
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument("--model_path", type=str, help="Path to model")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    args = parser.parse_args()

    # Run prediction
    predictions = predict(args.model_path, args.dataset_path)

    # Convert predictions to numpy for easy handling
    predictions_np = predictions.cpu().numpy()

    # Define label mapping
    label_map = {0: "not-hate", 1: "hate"}

    # Get the predicted labels
    predicted_labels = predictions_np.argmax(axis=1)

    # Save predictions and labels to a text file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    PREDICTION_PATH = os.path.join("outputs", "predictions", f"predictions_{current_time}.txt")

    os.makedirs(os.path.dirname(PREDICTION_PATH), exist_ok=True)

    with open(PREDICTION_PATH, "w") as f:
        f.write("Probability_Label_0, Probability_Label_1, Predicted_Label\n")
        for probs, label in zip(predictions_np, predicted_labels):
            f.write(f"{probs[0]:.4f}, {probs[1]:.4f}, {label_map[label]}\n")

    print(f"Predictions saved to {PREDICTION_PATH}")
