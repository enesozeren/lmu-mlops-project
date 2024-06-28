import numpy as np
import os


def read_dataset(tweets_file, labels_file=None):
    """
    Read the tweets and labels from the given files
    If labels_file is None, return only the tweets
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

    data_dir = "data/raw"

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


# Metrics
def _b_tp(preds, labels):
    """Returns True Positives (TP): count of correct predictions of actual class 1"""
    return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])


def _b_fp(preds, labels):
    """Returns False Positives (FP): count of wrong predictions of actual class 1"""
    return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])


def _b_tn(preds, labels):
    """Returns True Negatives (TN): count of correct predictions of actual class 0"""
    return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])


def _b_fn(preds, labels):
    """Returns False Negatives (FN): count of wrong predictions of actual class 0"""
    return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])


def b_metrics(preds, labels):
    """
    Returns the following metrics:
      - accuracy    = (TP + TN) / N
      - precision   = TP / (TP + FP)
      - recall      = TP / (TP + FN)
      - specificity = TN / (TN + FP)
    """
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    tp = _b_tp(preds, labels)
    tn = _b_tn(preds, labels)
    fp = _b_fp(preds, labels)
    fn = _b_fn(preds, labels)
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else "nan"
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else "nan"
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else "nan"
    return b_accuracy, b_precision, b_recall, b_specificity
