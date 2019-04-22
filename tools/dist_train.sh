#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CUDA_VISIBLE_DEVICES=1 $PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
