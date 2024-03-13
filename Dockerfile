FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
RUN apt-get update -y && \
    apt-get upgrade -y && \ 
    apt-get install -y python3-pip python3-dev openjdk-11-jdk
COPY . /opt
WORKDIR /opt
RUN pip3 install -r requirements.txt
RUN export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64


CMD ["bash"]
