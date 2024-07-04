# Base image
FROM europe-west3-docker.pkg.dev/lmumlops/hatespeechapi/hatespeech-base

# Set the working directory in the container
WORKDIR /lmu-mlops-project



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
