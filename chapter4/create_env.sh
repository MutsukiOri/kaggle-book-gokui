#! /bin/sh
set -eux

docker build -t kaggle-gokui-chapter4:latest .

docker run --gpus all -it --init \
 -v ${HOME}/.kaggle:/root/.kaggle \
 -v ${HOME}/.cache:/root/.cache \
 -v `pwd`/data:/workspace/data \
 -v `pwd`/code:/workspace/code \
 kaggle-gokui-chapter4 \
 bash