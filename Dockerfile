# syntax=docker/dockerfile:1

# This is our first build stage, it will not persist in the final image
FROM ubuntu as intermediate
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client && rm -rf /var/lib/apt/lists/*
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts
# Download the computer vision framework
RUN --mount=type=ssh git clone git@github.com:bobetocalo/dad22_landmarks.git dad22_landmarks
ADD data /dad22_landmarks/data

# Copy the repository from the previous image
FROM nvcr.io/nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ENV LANG=C.UTF-8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget cmake libgl1-mesa-glx libsm6 libxext6 libglib2.0-0
RUN mkdir -p /home/username
WORKDIR /home/username
COPY --from=intermediate /dad22_landmarks /home/username/dad22_landmarks
LABEL maintainer="roberto.valle@upm.es"
# Setup conda environment
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/username/miniconda.sh
RUN chmod +x /home/username/miniconda.sh
RUN /home/username/miniconda.sh -b -p /home/username/conda
RUN /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /home/username/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /home/username/conda/bin/conda create --name dad22 python=3.8
# Activate conda environment
ENV PATH /home/username/conda/envs/dad22/bin:/home/username/conda/bin:$PATH
# Make RUN commands use the new environment (source activate dad22)
SHELL ["conda", "run", "-n", "dad22", "/bin/bash", "-c"]
# Install dependencies
RUN conda install pytorch==1.9.0 torchvision==0.10.0 torchmetrics==0.11.4 torch-optimizer==0.1.0 pytorch-lightning==1.6.0 cudatoolkit=11.3 -c conda-forge
RUN pip install "mkl==2023.1.0" images-framework pytorch-toolbelt==0.5.0 coloredlogs albumentations==1.0.0 hydra-core==1.1.0 smplx==0.1.26
RUN pip install --force-reinstall "numpy==1.23.5"
RUN pip install --no-build-isolation chumpy==0.70 