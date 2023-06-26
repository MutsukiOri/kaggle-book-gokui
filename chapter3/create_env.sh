#!/bin/sh
set -eux

docker build  -t kaggle-gokui-chapter3:latest .


docker run --gpus all -it --init -v .:/home/chapter3 --memory-reservation="24gb" --shm-size=2g --name kaggle-gokui-chapter3 kaggle-gokui-chapter3:latest
