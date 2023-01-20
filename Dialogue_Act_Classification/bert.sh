#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-09:00     # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

#module load cuda/11.1
module load cuda cudnn 

#source tensorflow/bin/activate
#python text_classification.py #text_classification.py #bert_load.py #confusion_matrix.py #bert.py #tensorflow-test.py
#python bert_switchboard.py

#wandb login 2d63ea919ccfd327371db5c19744bc23b77def8b 
#python baseline_advanced/casa/main.py

python bert.py

#python baseline_advanced/bi-lstm-crf/core.py
#python text_classification.py
#python camem_bert.py

#python text_classification.py
#python gpt2.py