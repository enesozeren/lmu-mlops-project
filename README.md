# MLOps project description

## Goal
This is the project description for the Machine Learning Operations course (summer semester 2024) at Ludwig Maximilian University. The overall goal of the project is to classify hate speech tweets. We will be working with the Transformers framework, utilizing Open ELM models, and the TweetEval Dataset. The focus of the project work is to demonstrate the incorporation of machine learning operations tools.

## Framework
As we are confronted with a natural language processing [NLP] task, we decided to use the [Transformers](https://github.com/huggingface/transformers) framework. It provides a multitude of pretrained models for tasks like our text classification problem. Also, this framework is backed by PyTorch - our deep learning library of choice.
We plan to utilize the BERT (Bidirectional Encoder Represenations from Transformers) model [BERT](https://huggingface.co/docs/transformers/model_doc/bert) provided by Hugging Face.

## Data
The data we have chosen to work with is the [TweetEval Dataset](https://arxiv.org/pdf/2010.12421) which contains different tweets dataset for different NLP tasks. We will be working on the Hate Speech Detection dataset. It consists of tweets and labels (0: not-hate, 1: hate). The paper which offers this dataset also includes benchmarks for the best-performing models available at the time of publication.

## Models
Currently large language models has the state-of-the-art results for most NLP tasks. In this project, we will use BERT, which is known for its strong performance on various NLP tasks, including text classification. BERT uses a transformer-based architecture and has been pretrained on a large corpus of text. We will fine-tune BERT for the hate speech detection task using the TweetEval dataset.

# Repository

## Structure & Explanations
```
.
├── LICENSE
├── Makefile
├── README.md
├── api                                         <- Has scripts to create a FastAPI for inference
│   ├── __init__.py
│   └── main.py
├── cloudbuild                                  <- Directory for continious integration with GCP
├── data                                        <- Contains data and dvc file
│   ├── raw
│   └── raw.dvc
├── dockerfiles                                 <- Contains dockerfiles for training, prediction, api
│   ├── hatespeech_base.dockerfile
│   ├── inference_api.dockerfile
│   ├── predict_model.dockerfile
│   └── train_model.dockerfile
├── docs
│   ├── README.md
│   ├── mkdocs.yaml
│   └── source
├── mlops_project                               <- Source code directory
│   ├── __init__.py
│   ├── data
│   │   └── make_dataset.py                     <- To get the data from the original source
│   ├── hate_speech_model.py                    <- Model
│   ├── checkpoints                             <- Contains the trained model weigts (tracked with dvc)
│   ├── checkpoints.dvc
│   ├── predict_model.py                        <- Script for prediction with trained model weights
│   └── train_model.py                          <- Script for training
├── outputs
│   └── predictions                             <- Contains outputs from for predict_model.py script
├── pyproject.toml                              <- File for building environment
├── reports                                     <- Contains answers to LMU MLOps lecture questions
│   ├── README.md
│   ├── figures
│   └── report.py
├── requirements.txt                            <- requirements for inference
├── environment.yaml                            <- file to recreate the conda env
├── requirements_dev.txt                        <- requirements for development
├── tests                                       <- Contains unit tests and api load tests
│   ├── __init__.py
│   ├── api_performance_locustfile.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_hate_speech_model.py
│   ├── test_predict_model.py
│   └── test_utils.py
└── utils                                       <- Contains utility functions to be used in other scripts
    ├── __init__.py
    └── utils_functions.py
```

## Conda environment
To create conda environment with the requirements of this repository, simply use
```bash
make conda_environment
```

## Dataset
To get the dataset and trained model weights, use
```bash
dvc pull
```
Note: You need GCP bucket permissions to be able to run this command

## Inference
Predictions from this script are saved to outputs directory. To make a prediction, use
```bash
python mlops_project/predict_model.py \
--model_path=/your/model/path.txt \
--dataset_path=/your/data/path.txt
```

To run the inference api locally, use
```bash
uvicorn --port 8000 api.main:app
```

## Inference APIs
To use the api served by Google Cloud Platform you can use the following link

Welcome endpoint
```bash
https://hate-speech-detection-cloudrun-api-sjx4y77sda-ey.a.run.app
```

Prediction for one tweet end point
```bash
https://hate-speech-detection-cloudrun-api-sjx4y77sda-ey.a.run.app/predict_labels_one_tweet?tweet=this is my twwetttt
```

## Training
To train the model, specify a hyperparameter yaml file and use
```bash
python mlops_project/train_model.py --config=mlops_project/config/config-defaults-sweep.yaml
```

## Docker

### Building Docker Images

Please first build the base docker image before building train / predict / inference api docker images
```bash
docker build -f dockerfiles/hatespeech_base.dockerfile . -t hatespeech-base:latest
```

To build the docker image for inference api, use
```bash
docker build -f dockerfiles/inference_api.dockerfile . -t inference_api:latest
```

### Running Docker Containers

To run the docker image for inference api, use
```bash
docker run -p 8080:8080 -e PORT=8080 inference_api:latest
```

You can also use the predict_model docker image by mounting with your machine for your model weights and dataset
```bash
docker run -v /home/user/models:/container/models \
           -v /home/user/data:/container/data \
           predict_model:latest \
           --model_path /container/models/model.pth \
           --dataset_path /container/data/test_text.txt
```

## Tests

Unit tests for this repo can be found in the ``tests/`` directory.

To do the locust test for the api load test run the following command
```bash
locust -f tests/api_performance_locustfile.py
```
