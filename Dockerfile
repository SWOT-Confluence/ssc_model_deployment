FROM ubuntu:20.04 AS stage0

WORKDIR /opt
COPY requirements.txt /opt
RUN mkdir /opt/hydroshare

USER root

RUN apt-get update
# RUN apt install -y software-properties-common
# RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.9
RUN apt-get install -y python-dev
RUN apt-get install -y python3-pip 
RUN apt-get install -y wget 
RUN apt-get install -y build-essential 
RUN apt-get install -y software-properties-common 
RUN apt-get install -y apt-utils 
RUN apt-get install -y ffmpeg 
RUN apt-get install -y libsm6 
RUN apt-get install -y libxext6 
RUN apt-get install -y libgdal-dev
RUN apt-get install -y --allow-downgrades libpq5=12.2-4
RUN apt-get install -y libpq-dev 
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
RUN apt-get install -y aptitude
RUN aptitude install -y libpq-dev 


FROM stage0 AS stage1

RUN apt-get update
RUN pip3 install --upgrade pip

FROM stage1 AS stage2
RUN pip3 install -r requirements.txt

# install boto3 after the other packages
RUN pip3 install boto3

# Force reinstall and upgrade Fiona to version 1.9.2 (or later)
RUN pip3 uninstall -y fiona && pip3 install fiona==1.9.2

RUN ldconfig
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

FROM stage2 AS stage3
COPY . /opt/

LABEL version="1.0" \
        description="SSC Prediction Module" \
        "confluence.contact"="tsimmons@umass.edu" \
        "multitask_algorithm.contact"="rdaroya@umass.edu" \
        "ssc_prediction_algorithm.contact"="luisa.lucchese@pitt.edu"
ENTRYPOINT [ "/usr/bin/python3", "/opt/process_ssc.py" ]