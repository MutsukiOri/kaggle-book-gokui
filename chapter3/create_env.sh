#!/bin/sh
set -eux

docker build  -t kaggle-gokui-chapter3:latest .


docker run --gpus all -it --init -v .:/home/chapter3 --env LOCAL_UID=$(id -u $USER) --env LOCAL_GID=$(id -g $USER) --name kaggle-gokui-chapter3 -p 5$(id -u $USER):8888 kaggle-gokui-chapter3:latest