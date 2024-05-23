# MLOps project description

## Goal
This is the project description for the Machine Learning Operations course (summer semester 2024) at Ludwig Maximilian University. The overall goal of the project is to classify human and machine generated tweets for deep fake social media text detection. We will be working with the Transformers framework, utilizing Open ELM models, and the Twitter deep Fake text Dataset. The focus of the project work is to demonstrate the incorporation of machine learning operations tools.

## Framework
As we are confronted with a natural language processing [NLP] task, we decided to use the [Transformers](https://github.com/huggingface/transformers) framework. It provides a multitude of pretrained models for tasks like our text classification problem. Also, this framework is backed by PyTorch - our deep learning library of choice.
We plan to utilize the Open Efficient Language Models [OpenELM](https://huggingface.co/apple/OpenELM) provided by Hugging Face.

## Data
The data we have chosen to work with is the [TweepFake - Twitter deep Fake text Dataset](https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text/data) from Kaggle consisting of 25K tweets classified as either human or ai generated. It consists of 5 columns (user_id, status_id, screen_name, account.type, class_type). Additionally we are planning to generate our own tweets using a LLM different from our classifier model. To balance out the distribution we will also scrape known twitter personalities as genuine tweet data.

## Models
Currently large language models has the state-of-the-art results for most NLP tasks. In this project we will also use an LLM but we’ll focus on relatively smaller ones. One of the recent and well performing open source models are Open ELM models by Apple. These models use a layer-wise scaling strategy to allocate parameters within each layer of the transformer model. There are 4 pre-trained and instruction tuned models with different sizes: 270M, 450M, 1.1B and 3B. We will be doing experiments with the smallest 2 models.
