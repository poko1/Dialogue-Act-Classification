#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=1-00:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 

SRC=~/projects/def-zaiane/maliha/augmentation/src
CACHE=~/projects/def-zaiane/maliha/augmentation/CACHE
TASK=act
BERTLR=4e-5

for NUMEXAMPLES in 10;
do
    for i in {0..14}; ##{0..14};
        do
        RAWDATADIR=~/projects/def-zaiane/maliha/augmentation/src/utils/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

       # Baseline classifier
        #python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log
        ## running bert on original data for classification

        # #######################
        # # CBERT Classifier
        # #######################

        CBERTDIR=$RAWDATADIR/cbert
        mkdir $CBERTDIR
        python $SRC/bert_aug/cbert.py --data_dir $RAWDATADIR --output_dir $CBERTDIR --task_name $TASK  --num_train_epochs 10 --seed ${i}  --cache $CACHE > $RAWDATADIR/cbert.log
        cat $RAWDATADIR/train.tsv $CBERTDIR/cbert_aug.tsv > $CBERTDIR/train.tsv
        cp $RAWDATADIR/test.tsv $CBERTDIR/test.tsv
        cp $RAWDATADIR/dev.tsv $CBERTDIR/dev.tsv
        python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CBERTDIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_cbert.log
        
    done
done


