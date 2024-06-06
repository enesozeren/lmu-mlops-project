from data.utils import get_datasets

# Get the training, validation, and test datesets
(train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels) = get_datasets()