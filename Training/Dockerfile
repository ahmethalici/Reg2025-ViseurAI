FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenslide0 \
    libhdf5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt