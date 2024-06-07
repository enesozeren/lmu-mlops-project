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

To get the dataset, use 
```bash
dvc pull
```