#!/bin/zsh
#PBS -P ee
#PBS -l select=1:ngpus=1:ncpus=1
#PBS -N gafns1,15

MAIN_MODULE=/home/ee/btech/ee1160440/scratch/Deep-RL-Transfer3

export PYTHONPATH=$MAIN_MODULE:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils/gat:$PYTHONPATH

JOB_DIR=$MAIN_MODULE/multi_train/a3c-gat-fc-ne-se
cd $JOB_DIR

#miniconda3
export PATH=/home/ee/btech/ee1160440/miniconda3/bin:$PATH
source activate python2

python3 train.py --domain=sysadmin --num_instances=0 --parallelism=4 --instance=1,11,12,13,14,15 --num_features=3 --activation="lrelu" --lr=0.001 --neighbourhood=4
