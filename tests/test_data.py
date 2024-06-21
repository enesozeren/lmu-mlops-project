import os

def test_raw_data():
    # Check if there are necessary files in the data/raw directory
    assert os.path.exists('data/raw/mapping.txt'), "mapping.txt not found"
    assert os.path.exists('data/raw/train_labels.txt'), "train_labels.txt not found"
    assert os.path.exists('data/raw/val_labels.txt'), "val_labels.txt not found"
    assert os.path.exists('data/raw/val_text.txt'), "val_text.txt not found"
    assert os.path.exists('data/raw/test_labels.txt'), "test_labels.txt not found"
    assert os.path.exists('data/raw/test_text.txt'), "test_text.txt not found"    

def test_files_not_empty():
    # Check if the files are not empty
    with open('data/raw/train_labels.txt') as f:
        assert len(f.read()) > 0, "train_labels.txt is empty"
    with open('data/raw/train_text.txt') as f:
        assert len(f.read()) > 0, "train_text.txt is empty"
    with open('data/raw/val_labels.txt') as f:
        assert len(f.read()) > 0, "val_labels.txt is empty"
    with open('data/raw/val_text.txt') as f:
        assert len(f.read()) > 0, "val_text.txt is empty"
    with open('data/raw/test_labels.txt') as f:
        assert len(f.read()) > 0, "test_labels.txt is empty"
    with open('data/raw/test_text.txt') as f:
        assert len(f.read()) > 0, "test_text.txt is empty"

def test_label_text_match():
    # Check if the number of labels and text match
    with open('data/raw/train_labels.txt') as f:
        train_labels = f.readlines()
    with open('data/raw/train_text.txt') as f:
        train_text = f.readlines()
    assert len(train_labels) == len(train_text), "Number of train labels and text do not match"

    with open('data/raw/val_labels.txt') as f:
        val_labels = f.readlines()
    with open('data/raw/val_text.txt') as f:
        val_text = f.readlines()
    assert len(val_labels) == len(val_text), "Number of val labels and text do not match"

    with open('data/raw/test_labels.txt') as f:
        test_labels = f.readlines()
    with open('data/raw/test_text.txt') as f:
        test_text = f.readlines()
    assert len(test_labels) == len(test_text), "Number of test labels and text do not match"

def test_mapping():
    # Check if 0 corresponds to not-hate and 1 corresponds to hate
    with open('data/raw/mapping.txt') as f:
        mapping = f.readlines()
    assert len(mapping) == 2, "Mapping file should have 2 lines"
    assert mapping[0].strip() == '0\tnot-hate', "0 should correspond to not-hate"
    assert mapping[1].strip() == '1\thate', "1 should correspond to hate"
