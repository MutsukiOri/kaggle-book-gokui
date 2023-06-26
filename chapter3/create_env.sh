#!/bin/sh
set -eux

docker build  -t kaggle-gokui-chapter3:latest .


docker run --gpus all -it --init -v .:/home/chapter3 -v ${HOME}/.kaggle:/root/.kaggle --shm-size=2g --name kaggle-gokui-chapter3 kaggle-gokui-chapter3:latest
