import os
from contextlib import asynccontextmanager
from http import HTTPStatus

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_PATH = os.path.join("mlops_project", "checkpoints", "best-checkpoint.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        state_dict=torch.load(MODEL_PATH, map_location=device),
    )
    model.eval()

    # Set the model and tokenizer in the app state
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device
    print("Welcome! Model loaded successfully!")

    yield

    # Clean up the model
    del model
    del tokenizer
    print("Model cleaned up! Goodbye!")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Welcome to the Hate Speech Detection model API!"}


@app.post("/predict_labels_one_tweet")
async def predict_labels_one_tweet(tweet: str):
    """Predict the labels of a tweet being hate speech or not."""
    if not tweet:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Please provide a tweet.",
        )

    # Tokenize the tweet
    tokenizer = app.state.tokenizer
    inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move inputs to the device
    device = app.state.device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform prediction
    model = app.state.model
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        # Convert to labels
        label = torch.argmax(predictions, dim=1)

    # Map label to not-hate and hate
    label = "not-hate" if label == 0 else "hate"

    return label


@app.post("/predict_labels_tweets_file")
async def predict_labels_tweets_file(file: UploadFile = File(...)):
    """
    Predict the labels of tweets in a file being hate speech or not.
    Each tweet should be on a new line.
    """
    # Read the file
    contents = await file.read()
    tweets = contents.decode("utf-8").split("\n")

    # Tokenize the tweets
    tokenizer = app.state.tokenizer
    inputs = tokenizer(tweets, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move inputs to the device
    device = app.state.device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform prediction
    model = app.state.model
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        # Convert to labels
        labels = torch.argmax(predictions, dim=1)

    # Map labels to not-hate and hate
    labels = ["not-hate" if label == 0 else "hate" for label in labels]

    return labels
