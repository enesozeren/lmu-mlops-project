# Base image
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

RUN apt update && \
    apt install --no-install-recommends -y python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
