# Base image
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /lmu-mlops-project

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY data/ data/
COPY mlops_project/ mlops_project/
COPY utils/ utils/
COPY outputs/ outputs/

# Set environment variable
ENV PYTHONPATH=/lmu-mlops-project

# Do not set the directory to root
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_project/predict_model.py", \
"--model_path", "mlops_project/models/saved_models/bsc_weights.pth", \
"--dataset_path", "data/raw/test_text.txt"]
