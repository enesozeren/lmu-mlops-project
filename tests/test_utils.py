from unittest.mock import mock_open, patch

from transformers import BertTokenizer

from utils.utils_functions import get_datasets, preprocessing, read_dataset, tokenize_tweets


# Test function for read_dataset without labels
@patch("builtins.open", new_callable=mock_open, read_data="tweet1\ntweet2\ntweet3\n")
def test_read_dataset_without_labels(mock_file):
    tweets, labels = read_dataset("fake_tweets_file.txt")
    assert tweets == ["tweet1", "tweet2", "tweet3"]
    assert labels is None


# Test function for read_dataset with labels
@patch("builtins.open", new_callable=mock_open, read_data="0\n1\n0\n")
def test_read_dataset_with_labels(mock_file):
    mock_file.side_effect = [
        mock_open(read_data="tweet1\ntweet2\ntweet3\n").return_value,
        mock_open(read_data="0\n1\n0\n").return_value,
    ]
    tweets, labels = read_dataset("fake_tweets_file.txt", "fake_labels_file.txt")
    assert tweets == ["tweet1", "tweet2", "tweet3"]
    assert labels == [0, 1, 0]


# Test function for get_datasets
@patch("os.path.join", side_effect=lambda *args: "/".join(args))
@patch("utils.utils_functions.read_dataset")
def test_get_datasets(mock_read_dataset, mock_path_join):
    mock_read_dataset.side_effect = [
        (["train_tweet1", "train_tweet2"], [0, 1]),
        (["val_tweet1", "val_tweet2"], [1, 0]),
        (["test_tweet1", "test_tweet2"], [0, 1]),
    ]
    datasets = get_datasets()
    expected = (
        (["train_tweet1", "train_tweet2"], [0, 1]),
        (["val_tweet1", "val_tweet2"], [1, 0]),
        (["test_tweet1", "test_tweet2"], [0, 1]),
    )
    assert datasets == expected


# Test function for preprocessing
def test_preprocessing():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_text = "Hello, world!"
    encoding = preprocessing(input_text, tokenizer)
    assert "input_ids" in encoding
    assert "attention_mask" in encoding
    assert encoding["input_ids"].shape == (1, 160)
    assert encoding["attention_mask"].shape == (1, 160)


# Test function for tokenize_tweets
def test_tokenize_tweets():
    tweets = ["Hello, world!", "Another tweet"]
    token_ids, attention_masks = tokenize_tweets(tweets)
    assert token_ids.shape == (2, 160)
    assert attention_masks.shape == (2, 160)
