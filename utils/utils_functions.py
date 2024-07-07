import os

import torch
from transformers import BertTokenizer


def read_dataset(tweets_file, labels_file=None):
    """
    Read the tweets and labels from the given files
    If labels_file is None, returns tweets and None
    Returns a tuple of lists in the following order: (tweets, labels)
    """

    # Step 1: Read the Text Files
    with open(tweets_file, "r") as f:
        tweets = f.readlines()

    tweets = [tweet.strip() for tweet in tweets]

    if labels_file is None:
        return tweets, None

    with open(labels_file, "r") as f:
        labels = f.readlines()

    labels = [int(label.strip()) for label in labels]

    return tweets, labels


def get_datasets():
    """
    Get the training, validation and test sets
    Returns a tuple of tuples in the following order: ((train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels))
    """

    data_dir = "data/raw" if len(os.listdir("/gcs")) == 0 else "/gcs/data/raw"

    # Load train, validation, and test datasets
    train_tweets, train_labels = read_dataset(
        os.path.join(data_dir, "train_text.txt"), os.path.join(data_dir, "train_labels.txt")
    )
    val_tweets, val_labels = read_dataset(
        os.path.join(data_dir, "val_text.txt"), os.path.join(data_dir, "val_labels.txt")
    )
    test_tweets, test_labels = read_dataset(
        os.path.join(data_dir, "test_text.txt"), os.path.join(data_dir, "test_labels.txt")
    )

    return (train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels)


# Preprocessing
def preprocessing(input_text, tokenizer):
    """
    Returns <class transformers.tokenization_utils_base.BatchEncoding>
    tokenizer.encode_plus() converts sequences into input formats for Bert-based classifier
    It returns a dictionary of three objects:
      - input_ids: list of token id
      - token_type_ids: list of ids indicate the sentence number that tokens belong to
      - attention_mask: list of indices (0,1) specifying which tokens should be considered by the model (return_attention_mask = True).
    """
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,  # adds [SEP], [CLS]
        max_length=160,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_tweets(tweets):
    token_ids = []
    attention_masks = []
    for sample in tweets:
        encoding_dict = preprocessing(sample, tokenizer)
        token_ids.append(encoding_dict["input_ids"])
        attention_masks.append(encoding_dict["attention_mask"])
    token_ids = torch.cat(token_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return token_ids, attention_masks
