# Base image
FROM hatespeech-base

# Set the working directory in the container
WORKDIR /lmu-mlops-project

# Copy the necessary files
COPY api/ api/
COPY mlops_project/checkpoints mlops_project/checkpoints

# Set environment variable
ENV PYTHONPATH=/lmu-mlops-project

# Make port available to the world outside this container
EXPOSE $PORT

# Run the command to start uWSGI with the specified port and host
ENTRYPOINT ["uvicorn", "api.main:app", "--port", "$PORT", "--host", "0.0.0.0"]
