#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=01-00:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 

SRC=~/projects/def-zaiane/maliha/data_augmentation/src
CACHE=~/projects/def-zaiane/maliha/data_augmentation/CACHE
TASK=da_100
BERTLR=4e-5

for NUMEXAMPLES in 100;
do
    for i in {0..14}; ##{0..14};
        do
        RAWDATADIR=~/projects/def-zaiane/maliha/data_augmentation/src/utils/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

       # Baseline classifier
        #python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log
        ## running bert on original data for classification

        #######################
        # GPT2 Classifier
        #######################

        GPT2DIR=$RAWDATADIR/gpt2
        mkdir $GPT2DIR
        python $SRC/bert_aug/cgpt2_edit.py --data_dir $RAWDATADIR --output_dir $GPT2DIR --task_name $TASK  --num_train_epochs 25 --seed ${i} --top_p 0.9 --temp 1.0 --cache $CACHE
        cat $RAWDATADIR/train.tsv $GPT2DIR/cmodgpt2_aug_3.tsv > $GPT2DIR/train.tsv
        cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
        cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
        python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $GPT2DIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_gpt2_3.log
    
    done
done


