# Base image
FROM hatespeech-base

# Set the working directory in the container
WORKDIR /lmu-mlops-project

# Copy required files
COPY pyproject.toml pyproject.toml
COPY data/ data/
COPY mlops_project/ mlops_project/
COPY utils/ utils/
COPY outputs/ outputs/

# Set environment variable
ENV PYTHONPATH=/lmu-mlops-project

# Set the entrypoint to the python script
ENTRYPOINT ["python3", "-u", "mlops_project/predict_model.py"]

# Provide default arguments that can be overridden
CMD ["--model_path", "mlops_project/checkpoints/best-checkpoint.pth", \
     "--dataset_path", "data/raw/test_text.txt"]
