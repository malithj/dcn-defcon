FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# update packages
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=US/Eastern apt-get install tzdata -y

# install essential tools
RUN apt-get install wget -y
RUN apt-get install git -y
RUN apt-get install curl -y

# install anaconda
RUN cd /tmp/ && \
    curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    echo "export PATH=$PATH:/opt/anaconda3/bin" >> ~/.bashrc 

# set path
ENV PATH "${PATH}:/opt/anaconda3/bin"

# shell
SHELL ["/bin/bash", "-c"]

# copy src
COPY . /home/dcn

# install anaconda environments
RUN eval "$(/opt/anaconda3/bin/conda shell.bash hook)" && \
    cd /home/dcn/ && \
    conda env create --name dcn -f environment.yml

# install dependencies
RUN eval "$(/opt/anaconda3/bin/conda shell.bash hook)" && \
    cd /home/dcn && \
    conda activate dcn
