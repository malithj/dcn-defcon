FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# update packages
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=US/Eastern apt-get install tzdata -y

# install essential tools
RUN apt-get install wget -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install python3.8 -y
RUN apt-get install python3.8-venv -y
RUN apt-get install python3.8-dev -y
RUN apt-get install python-is-python3 -y

# copy src
COPY . /home/dcn

# activate shell
SHELL ["/bin/bash", "-c"] 

# install dependencies
RUN cd /home/dcn && \
    python -m venv /home/dcn/venv && \
    source venv/bin/activate && \ 
    pip install -r requirements.txt
