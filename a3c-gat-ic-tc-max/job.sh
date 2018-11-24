#!/bin/zsh

JOB_DIR=/home/ee/btech/ee1160440/scratch/Deep-RL-Transfer3/multi_train/a3c-gat-ic-tc-max
cd $JOB_DIR

# qsub -P ee -l select=1:ngpus=1:ncpus=1 sysadmin.sh
qsub -P ee -l select=1:ngpus=1:ncpus=1 gameoflife.sh