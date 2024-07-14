# Base image
FROM hatespeech-base

WORKDIR /lmu-mlops-project

COPY pyproject.toml pyproject.toml
COPY mlops_project/ mlops_project/
COPY utils/ mlops_project/utils/
COPY data/ data/

ENTRYPOINT ["python3", "-u", "mlops_project/train_model.py"]
