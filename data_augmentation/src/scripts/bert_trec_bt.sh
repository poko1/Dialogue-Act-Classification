#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=1-00:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 

SRC=~/projects/def-zaiane/maliha/data_augmentation/src
CACHE=~/projects/def-zaiane/maliha/data_augmentation/CACHE
TASK=da_100  #new_old #imdb_100 #emotion_whole #emotion_20 #imdb_20
BERTLR=4e-5

for NUMEXAMPLES in 100;
do
    for i in {0..14}; ##{0..14}; #9-14 for emo 20
        do
        RAWDATADIR=~/projects/def-zaiane/maliha/data_augmentation/src/utils/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

       # Baseline classifier
        #python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log
        ## running bert on original data for classification

    #    #######################
    #    # Backtranslation DA Classifier
    #    #######################

        BTDIR=$RAWDATADIR/bt
        mkdir $BTDIR
        python $SRC/bert_aug/backtranslation.py --data_dir $RAWDATADIR --output_dir $BTDIR --task_name $TASK  --seed ${i} --cache $CACHE
        cat $RAWDATADIR/train.tsv $BTDIR/bt_aug.tsv > $BTDIR/train.tsv #da_10_bt
        cp $RAWDATADIR/test.tsv $BTDIR/test.tsv
        cp $RAWDATADIR/dev.tsv $BTDIR/dev.tsv
        python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $BTDIR --seed ${i} --cache $CACHE  > $RAWDATADIR/bert_bt.log

    done
done


