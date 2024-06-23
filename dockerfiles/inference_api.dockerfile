# Base image
FROM python:3.12.3-slim

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

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the command to start uWSGI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]