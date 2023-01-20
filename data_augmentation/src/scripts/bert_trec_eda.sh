#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-01:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID


module load cuda cudnn

SRC=~/projects/def-zaiane/maliha/data_augmentation/src
CACHE=~/projects/def-zaiane/maliha/data_augmentation/CACHE
TASK=da_100 #new_old #six_emo #emotion_whole
BERTLR=4e-5

#i=0
#NUMEXAMPLES=0/home/maliha/projects/def-zaiane/maliha/data_augmentation/src/utils/datasets/six_emo
#RAWDATADIR=~/projects/def-zaiane/maliha/augmentation/src/utils/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

#python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log


for NUMEXAMPLES in 100;
do
    for i in {0..14}; ##{0..14};
        do
        RAWDATADIR=~/projects/def-zaiane/maliha/data_augmentation/src/utils/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

       # Baseline classifier
        #python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log
        ## running bert on original data for classification

      ##############
      ## EDA
      ##############

      EDADIR=$RAWDATADIR/eda
      
      mkdir $EDADIR  ## creates an eda directory for each segment of the dataset
      
      python $SRC/bert_aug/eda.py --input $RAWDATADIR/train.tsv --output $EDADIR/eda_aug.tsv --num_aug=1 --alpha=0.1 --seed ${i} ## applies EDA on train data segment to generate ada_aug
      
      cat $RAWDATADIR/train.tsv $EDADIR/eda_aug.tsv > $EDADIR/train.tsv ### ori + aug data in new dataset called train.csv under eda dir of each data segment
      
      cp $RAWDATADIR/test.tsv $EDADIR/test.tsv ## copy ori test file to eda folder
      
      cp $RAWDATADIR/dev.tsv $EDADIR/dev.tsv ## copy ori dev test file to eda folder
      
      python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $EDADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE  > $RAWDATADIR/bert_eda.log
      ## applies bert classifier on the 3 sets of data under eda folder of each data segment

    done
done


