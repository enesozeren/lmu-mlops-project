# Base image
FROM hatespeech-base


COPY pyproject.toml pyproject.toml
COPY mlops_project/ mlops_project/
COPY data/ mlops_project/data/
COPY utils/ mlops_project/utils/



WORKDIR /mlops_project

ENTRYPOINT wandb login && python -u train_model.py
