# Base image
FROM hatespeech-base

# Copy the necessary files
COPY api/ api/
COPY mlops_project/models mlops_project/models

# Set the working directory
WORKDIR /


# Make port available to the world outside this container
EXPOSE $PORT

# Run the command to start uWSGI with the specified port and host
CMD exec uvicorn api.main:app --port $PORT --host 0.0.0.0
