FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ADD requirements.txt /home/
WORKDIR /home/
RUN pip install -r requirements.txt

WORKDIR /home/chapter3
