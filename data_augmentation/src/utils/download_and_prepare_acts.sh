#!/usr/bin/env bash

#mkdir datasets
NUMDEV=10 #10
NUMEXP=15

## need to remove tabs within sentences

python create_fsl_dataset.py -datadir datasets/da_100 -num_train 100 -num_dev $NUMDEV -sim $NUMEXP -lower

## for every label: 10 python create_fsl_dataset.py -datadir datasets/trec -num_train 10 -num_dev $NUMDEV -sim $NUMEXP -lower

