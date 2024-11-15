[![python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

# Knowledgeable Goal Modeling

We provide the code used for our paper "An Interactive Tool for Goal Model Construction using a Knowledge Graph", written by Shahin ABDOUL SOUKOUR, William ABOUCAYA and Nikolaos GEORGANTAS.

## Installing the dependencies

Inside a dedicated Python environment:

```shell
pip install -r requirements.txt
```

## Run the tool

```shell
uvicorn main:app
```

The tool is then accessible by opening a webpage at the URL [127.0.0.1:8000](http://127.0.0.1:8000) or [localhost:8000](http://localhost:8000)

## Increasing the speed of the inferences using HuggingFace Inference Endpoints

If you use HuggingFace Inference Endpoints, you can perform the NLI and sentiment analysis tasks on remote servers by creating a `.env` file at the root of this project and adding the following environment variables:

- `HF_TOKEN`: Your HuggingFace Inference Endpoints access token
- `API_URL_NLI`: The URL to your endpoint containing a model dedicated to NLI (in our paper, we use `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`)
- `API_URL_SENT`: The URL to your endpoint containing a model dedicated to sentiment analysis (in our paper, we use `cardiffnlp/twitter-roberta-base-sentiment-latest`)
