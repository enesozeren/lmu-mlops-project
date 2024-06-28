# Base image
FROM python:3.12-slim

# install gcc and python3-dev
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the necessary files
COPY requirements.txt requirements.txt
COPY api/ api/
COPY mlops_project/models mlops_project/models

# Set the working directory
WORKDIR /

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# Make port available to the world outside this container
EXPOSE $PORT

# Run the command to start uWSGI with the specified port and host
CMD exec uvicorn api.main:app --port $PORT --host 0.0.0.0
