FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && \ 
    apt-get upgrade -y && \
    apt-get install ffmpeg libsm6 libxext6 git curl -y
RUN pip install --upgrade pip

RUN pip install ultralytics
COPY . .

COPY ./requirements.txt /app/
RUN pip install -r /app/requirements.txt
RUN rm /app/requirements.txt

WORKDIR /