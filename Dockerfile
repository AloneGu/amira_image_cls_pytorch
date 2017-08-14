FROM ubuntu:14.04
MAINTAINER jackling


RUN DEBIAN_FRONTEND=noninteractive

ENV LANG=en_US.UTF-8
ENV TIME_ZONE=Asia/Shanghai

# Update apt-get local index
RUN apt-get -qq update

# install pip and other dev
RUN apt-get -y install python3-pip python3-dev libxml2-dev libxslt-dev python-lxml libblas-dev liblapack-dev libatlas-base-dev gfortran libhdf5-dev

# add for pillow
RUN apt-get install libjpeg8-dev zlib1g-dev libfreetype6-dev

RUN pip3 install --upgrade pip

COPY . /app/

WORKDIR /app

# install python lib
RUN pip3 install --no-cache-dir -i https://pypi.doubanio.com/simple --extra-index-url https://pypi.gizwits.com:1443/root/gizwits -r requirments.txt

# use 0.1.12 version
RUN pip3 install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl  -i https://pypi.doubanio.com/simple
RUN pip3 install torchvision -i https://pypi.doubanio.com/simple

RUN mkdir -p /data/log && \
    rm -rf /tmp/* ~/.cache

RUN mkdir /root/.keras/
COPY keras.json /root/.keras/keras.json
RUN cat /root/.keras/keras.json

VOLUME ["/data/img_dir","/app/models"]
EXPOSE 8090
ENTRYPOINT ["/bin/sh", "--", "/app/entrypoint.sh"]