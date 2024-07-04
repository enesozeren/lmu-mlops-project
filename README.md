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

## Dataset
To get the dataset, use
```bash
dvc pull
```

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

## Training
TBD

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

(OR) To make sure your image has amd64 architecture (necessary for google cloud), you can use:
```bash
docker buildx build --platform linux/amd64 -f dockerfiles/inference_api.dockerfile . -t inference_api:latest
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
