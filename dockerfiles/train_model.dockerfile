# Base image
FROM europe-west3-docker.pkg.dev/lmumlops/hatespeechapi/hatespeech-base

COPY pyproject.toml pyproject.toml
COPY mlops_project/ mlops_project/
COPY data/ data/

WORKDIR /
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_project/train_model.py"]
