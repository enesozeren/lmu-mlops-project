# Base image
FROM hatespeech-base

COPY pyproject.toml pyproject.toml
COPY mlops_project/ mlops_project/
COPY utils/ mlops_project/utils/
COPY data/ data/

WORKDIR /mlops_project
ENTRYPOINT ["python3", "-u", "train_model.py"]
