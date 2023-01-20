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

        #Baseline classifier
        #python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log
        ## running bert on original data for classification

        # #######################
        # # CMODBERT Classifier
        # ######################

        CMODBERTDIR=$RAWDATADIR/cmodbert
        mkdir $CMODBERTDIR
        python $SRC/bert_aug/cmodbert.py --data_dir $RAWDATADIR --output_dir $CMODBERTDIR --task_name $TASK  --num_train_epochs 150 --learning_rate 0.00015 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbert.log
        cat $RAWDATADIR/train.tsv $CMODBERTDIR/cmodbert_aug.tsv > $CMODBERTDIR/train.tsv
        cp $RAWDATADIR/test.tsv $CMODBERTDIR/test.tsv
        cp $RAWDATADIR/dev.tsv $CMODBERTDIR/dev.tsv
        python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CMODBERTDIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_cmodbert.log

    done
done


