# Improving Dialogue Act Classification with Text Augmentation Techniques

This is part of my Master's Thesis on applying data augmentation techniques for improving dialogue act detection in chatbots. Dialogue act classification is the task of classifying an utterance with respect to the function it serves in a dialogue. Based on literature surveys and how our users correspond with two of out chatbots- MIRA and ANA, we first propose a dialogue act scheme with 8 dialogue acts:

![da](https://user-images.githubusercontent.com/42430946/213807350-8e70e9bd-f461-49f4-badd-4f743ab301db.PNG)

Then we curate a dataset with 24k user utterances (Dialogue_Act_Classification/train_da_10.csv and Dialogue_Act_Classification/test_da_10.csv) and annotate them with the proposed dialogue acts. We then train two models (baseline SVM and finetuned Bert) on the train dataset and achieve an accuracy rate of 97% and 99% respectively. To show the our model generalizes well on datasets created from completely unseen sources, we create a new test dataset (Dialogue_Act_Classification/dialogrando.csv). Upon running our trained models on this new test dataset, we still achieve an accuracy of 86% (SVM) and 96% (BERT). Thus proving that our models are actually learning the features from the training dataset and not memorizing the data itself.  

![Picture1](https://user-images.githubusercontent.com/42430946/213808465-902940fe-e184-4a85-b7bb-792bc4460a13.png)

We then try to improve the accuracy rate of our baseline model for the two minority classes: Greeting and Feedback by using text augmentation techniques. We used EDA (Synonym Replacement, Random Deletion of Word, Random Insertion of Word, Random Swap between two words) as well as Backtranslation (Translating existing example from English to German and then back to English) to generate new examples for Greeting and Feedback class from the existing examples and added them to the training data. Upon training our model with the newly added augmented dataset, we saw an increase in accuracy for both the classes (Greeting: from 88% to 93% with EDA and to 92% with BT, Feedback: from 87% to 89% with EDA).

To further understand how effective text data augmentation techniques can be on improving dialogue act classification, we created a low resource setting i.e only taking 10 examples per label so a total of 180 examples as our training dataset. We apply a number of data augmentation techniques like EDA, Back Translation, BERT Masked Token Prediction, GPT-2 Text Generation and so on to create one to three new example(s) from an existing example. After adding them to the low resource training dataset, we train our models and compare the performance. From the experiments, it can be concluded that in cases where there is data scarcity, almost all the data augmentation techniques can help improve the performance of the classifiers significantly (accuracy jumps from 76% to 83%). An exception here is Back Translation can sometimes generate new sentences that are no longer label preserving). We repeat the same experiment with 100 examples/label and draw similar conclusions (accuracy jumps from 88% to 96%).

![picture2](https://user-images.githubusercontent.com/42430946/213809510-9aae5594-124c-4305-8294-29ac468ce5c5.PNG)

# Requirements:
To run this code, you need following dependencies
- Pytorch 1.5
- fairseq 0.9
- transformers 2.9

# How to Run:
## Dialogue Act Classification:
In Dialogue_Act_Classification folder, run <code>sbatch bert.sh</code> to use bert-based model and run <code>python svm.py</code> to use SVM model
## Data Augmentation (Low-data regime experiment):
Run <code>src/utils/download_and_prepare_datasets.sh</code> file to prepare all datsets. For a given dataset, it creates 15 random splits of train and dev data.
To run data augmentation experiment for a given dataset, run bash script in scripts folder. For example: to run eda data augmentation,
run <code>scripts/bert_trec_eda.sh</code>

