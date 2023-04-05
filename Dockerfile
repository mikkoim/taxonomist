FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

COPY environment.yml .

RUN apt-get update --fix-missing && \
    apt-get install -y git wget vim tmux unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
    /bin/bash Miniconda.sh -b -p /opt/conda && \
    rm Miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    /opt/conda/bin/conda install -n base -c conda-forge mamba && \
    /opt/conda/bin/mamba update -n base mamba && \
    /opt/conda/bin/mamba env create --file environment.yml &&\
    /opt/conda/bin/mamba clean -a -y

ENV PATH /opt/conda/bin:$PATH
