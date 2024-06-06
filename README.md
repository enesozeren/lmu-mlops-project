# MLOps project description

## Goal
This is the project description for the Machine Learning Operations course (summer semester 2024) at Ludwig Maximilian University. The overall goal of the project is to classify hate speech tweets. We will be working with the Transformers framework, utilizing Open ELM models, and the TweetEval Dataset. The focus of the project work is to demonstrate the incorporation of machine learning operations tools.

## Framework
As we are confronted with a natural language processing [NLP] task, we decided to use the [Transformers](https://github.com/huggingface/transformers) framework. It provides a multitude of pretrained models for tasks like our text classification problem. Also, this framework is backed by PyTorch - our deep learning library of choice.
We plan to utilize the Open Efficient Language Models [OpenELM](https://huggingface.co/apple/OpenELM) provided by Hugging Face.

## Data
The data we have chosen to work with is the [TweetEval Dataset](https://arxiv.org/pdf/2010.12421) which contains different tweets dataset for different NLP tasks. We will be working on the Hate Speech Detection dataset. It consists of tweets and labels (0: not-hate, 1: hate). The paper which offers this dataset also includes benchmarks for the best-performing models available at the time of publication.

## Models
Currently large language models has the state-of-the-art results for most NLP tasks. In this project we will also use an LLM but we’ll focus on relatively smaller ones because current research trends emphasize developing small but high-performing models instead of very large models for various tasks. We want to test one of these small models, OpenELM Models by Apple, which are a recently developed and well performing open source models. These models use a layer-wise scaling strategy to allocate parameters within each layer of the transformer model. There are 4 pre-trained and instruction tuned models with different sizes: 270M, 450M, 1.1B and 3B. We will be doing experiments with the smallest 2 models.

# Repository

To get the dataset, use 
```bash
dvc pull
```