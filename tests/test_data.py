import os

def test_raw_data():
    # Check if there are necessary files in the data/raw directory
    assert os.path.exists('data/raw/mapping.txt'), "mapping.txt not found"
    assert os.path.exists('data/raw/test_labels.txt'), "test_labels.txt not found"
    assert os.path.exists('data/raw/test_text.txt'), "test_text.txt not found"
    assert os.path.exists('data/raw/train_labels.txt'), "train_labels.txt not found"
    assert os.path.exists('data/raw/val_labels.txt'), "val_labels.txt not found"
    assert os.path.exists('data/raw/val_text.txt'), "val_text.txt not found"