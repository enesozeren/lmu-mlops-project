def _read_dataset(tweets_file, labels_file):
    '''
    Read the tweets and labels from the given files
    Returns a tuple of lists in the following order: (tweets, labels)
    '''

    # Step 1: Read the Text Files
    with open(tweets_file, 'r') as f:
        tweets = f.readlines()

    with open(labels_file, 'r') as f:
        labels = f.readlines()

    tweets = [tweet.strip() for tweet in tweets]
    labels = [int(label.strip()) for label in labels]

    return tweets, labels


def get_datasets():
    '''
    Get the training, validation and test sets
    Returns a tuple of tuples in the following order: ((train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels))
    '''

    # Load train, validation, and test datasets
    train_tweets, train_labels = _read_dataset('data/raw/train_text.txt', 'data/raw/train_labels.txt')
    val_tweets, val_labels = _read_dataset('data/raw/val_text.txt', 'data/raw/val_labels.txt')
    test_tweets, test_labels = _read_dataset('data/raw/test_text.txt', 'data/raw/test_labels.txt')

    return (train_tweets, train_labels), (val_tweets, val_labels), (test_tweets, test_labels)
