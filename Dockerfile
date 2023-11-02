FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get upgrade -y && \ 
    apt-get install -y python3-pip python3-dev
COPY . /opt
WORKDIR /opt
RUN pip3 install -r requirements.txt


CMD ["bash"]
