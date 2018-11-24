#!/bin/zsh

MAIN_MODULE=$(pwd)/..

export PYTHONPATH=$MAIN_MODULE:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils/gat:$PYTHONPATH

JOB_DIR=$MAIN_MODULE/a3c-gat-ic-tc-max
cd $JOB_DIR

python3 train.py --domain=sysadmin --num_instances=0 --parallelism=4 --instance=$1 --num_features=3 --activation="lrelu" --lr=0.001 --neighbourhood=2
