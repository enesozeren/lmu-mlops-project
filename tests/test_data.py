import os
import pytest

DATA_DIR = "data/raw"


@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="Data files not found")
def test_raw_data():
    assert os.path.exists(os.path.join(DATA_DIR, "mapping.txt")), "mapping.txt not found"
    assert os.path.exists(os.path.join(DATA_DIR, "train_labels.txt")), "train_labels.txt not found"
    assert os.path.exists(os.path.join(DATA_DIR, "val_labels.txt")), "val_labels.txt not found"
    assert os.path.exists(os.path.join(DATA_DIR, "val_text.txt")), "val_text.txt not found"
    assert os.path.exists(os.path.join(DATA_DIR, "test_labels.txt")), "test_labels.txt not found"
    assert os.path.exists(os.path.join(DATA_DIR, "test_text.txt")), "test_text.txt not found"


@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="Data files not found")
def test_files_not_empty():
    assert_file_not_empty(os.path.join(DATA_DIR, "train_labels.txt"), "train_labels.txt is empty")
    assert_file_not_empty(os.path.join(DATA_DIR, "train_text.txt"), "train_text.txt is empty")
    assert_file_not_empty(os.path.join(DATA_DIR, "val_labels.txt"), "val_labels.txt is empty")
    assert_file_not_empty(os.path.join(DATA_DIR, "val_text.txt"), "val_text.txt is empty")
    assert_file_not_empty(os.path.join(DATA_DIR, "test_labels.txt"), "test_labels.txt is empty")
    assert_file_not_empty(os.path.join(DATA_DIR, "test_text.txt"), "test_text.txt is empty")


@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="Data files not found")
def test_label_text_match():
    assert_label_text_match(
        os.path.join(DATA_DIR, "train_labels.txt"),
        os.path.join(DATA_DIR, "train_text.txt"),
        "Number of train labels and text do not match",
    )
    assert_label_text_match(
        os.path.join(DATA_DIR, "val_labels.txt"),
        os.path.join(DATA_DIR, "val_text.txt"),
        "Number of val labels and text do not match",
    )
    assert_label_text_match(
        os.path.join(DATA_DIR, "test_labels.txt"),
        os.path.join(DATA_DIR, "test_text.txt"),
        "Number of test labels and text do not match",
    )


@pytest.mark.skipif(not os.path.exists(DATA_DIR), reason="Data files not found")
def test_mapping():
    assert_mapping(os.path.join(DATA_DIR, "mapping.txt"))


def assert_file_not_empty(file_path, error_message):
    with open(file_path) as f:
        assert len(f.read()) > 0, error_message


def assert_label_text_match(labels_path, text_path, error_message):
    with open(labels_path) as f_labels, open(text_path) as f_text:
        labels = f_labels.readlines()
        text = f_text.readlines()
        assert len(labels) == len(text), error_message


def assert_mapping(mapping_path):
    with open(mapping_path) as f:
        mapping = f.readlines()
    assert len(mapping) == 2, "Mapping file should have 2 lines"
    assert mapping[0].strip() == "0\tnot-hate", "0 should correspond to not-hate"
    assert mapping[1].strip() == "1\thate", "1 should correspond to hate"
