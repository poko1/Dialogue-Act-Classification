## Implementation of Pretrained BART using hugging face transformers library
## https://www.analyticsvidhya.com/blog/2021/12/multiclass-classification-using-transformers/
## https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
## https://www.tensorflow.org/tutorials/keras/save_and_load 
# importing the dataset 

# apologizing = ['sorry', 'pardon', 'my bad', 'apologize', 'apology', 'forgive me'] #'my mistake' sorry, what do u mean



import pandas as pd
import numpy as np
import torch
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer,TFBertModel
from sklearn.metrics import classification_report
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense 
#new_test.csv data/train_da_10.csv data/test_da_10.csv
#print('k_o')
df_train = pd.read_csv("/home/maliha/projects/def-zaiane/maliha/data/train_da_10.csv", header =None, sep =',', names = ['text', 'plabel','label'], encoding='mac_roman', skiprows=[0]) #utf-8
df_test = pd.read_csv("/home/maliha/projects/def-zaiane/maliha/data/test_da_10.csv", header = None, sep =',', names = ['text', 'plabel', 'label'],encoding='mac_roman', skiprows=[0])
#df_test = 'Alexa, how are you doing today?' 

df_train = df_train.dropna()
df_test = df_test.dropna()

###################################### 45 for bert_new_and_old
import string

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
        text = text.lower()
    return text

l = []

for i in df_train.text:
    i = remove_punctuations(i)
    l.append(i)
    
df_train.text = l

l = []

for i in df_test.text:
    i = remove_punctuations(i)
    l.append(i)
    
df_test.text = l
################################ 
#'A','GC','DD','QF','DI','FN','GO','FP','SS','QYN'

#['A','DD','QF','G','DI','F','S','QYN']

encoded_dict = {'a':0, 'd':1, 'f':2, 'g':3, 'i':4, 'pn':5, 's':6, 'yn':7}  #{'d':0, 'q':1, 's':2} {'d':0, 'r':1, 'yn':2, 'f':3, 'is':4, 'nis':5} {'d':0, 'i':1, 'yn':2, 'f':3, 's':4} 
#encoded_dict = {'i':0, 'q':1, 's':2}
df_train['label'] = df_train.label.map(encoded_dict)
df_test['label'] = df_test.label.map(encoded_dict)

################################


y_train = to_categorical(df_train.label)
y_test = to_categorical(df_test.label)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')#"bert-base-multilingual-uncased" bert-base-cased
bert = TFBertModel.from_pretrained('bert-base-cased')
#CUDA_VISIBLE_DEVICES=""

# here tokenizer using from bart-large
max_len = 70

x_train = tokenizer(
    text=df_train.text.tolist(),
    #add_special_tokens=True,
    padding="max_length",
    max_length=max_len, #25
    truncation=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

    
x_test = tokenizer(
    text=df_test.text.tolist(), #df_test,#
    #add_special_tokens=True,
    padding="max_length",
    max_length=max_len, #25
    truncation=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

#input_ids = x_train['input_ids']
#attention_mask = x_train['attention_mask']


#padding=max_length
#max_len = 70 #25 for kevin
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")#, padding=True)
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")#, padding=True)
embeddings = bert(input_ids,attention_mask = input_mask)[0] 
print(embeddings)
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(8,activation = 'sigmoid')(out)#3
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True #

### Model Compilation
optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

##########################################

import os

#checkpoint_path = "/home/maliha/projects/def-zaiane/maliha/bert_checkpoints_speechact"
#os.makedirs(checkpoint_path)

############

train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    validation_data = (
    {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
    ),
  epochs= 3, #3, 
  batch_size= 36 #36
  #callbacks=[cp_callback]
)


#bert.save_pretrained(checkpoint_path)

#model.save_weights(checkpoint_path) #now trying without old imp

#model.load_weights(checkpoint_path)

### Model Evaluation
#print(x_test['input_ids'])
#print(x_test['attention_mask'])
#'''
"""

import torch
import torch.nn.functional as F
logits = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
tensor_b = torch.Tensor(logits)
probabilities = F.softmax(tensor_b, dim=-1)

print(probabilities)
l = 0
m = 0
print('other')
for i in probabilities:
    a = i.tolist()
    m = max(a)
    if m<0.2: 
        print(df_test.text[l])
        #print('other')
    l = l+1

"""

predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
print(predicted_raw[0])


y_predicted = (np.argmax(predicted_raw, axis = 1)) #-1



y_true = df_test.label


print(classification_report(y_true, y_predicted))

# print index of wrongly predicted
print("'''''''")



for idx, input in df_test.iterrows(): 
    if y_predicted[idx] != y_true[idx]:
        print("No.", idx, 'input,',input.text, ', has been classified as', y_predicted[idx], 'and should be', y_true[idx]) 



confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)

#confusion_matrix = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=1)
confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
#['A','DD','QF','G','DI','F','S','QYN']   ['I','Q','S']
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['A','DD','QF','G','DI','F','S','QYN']) #stat ques imp ['D', 'Q', 'S']   ['D_D','Q_F','S_IS','S_NIS','D_R','Q_YN']
c = cm_display.plot()
s = 'bert_8.jpg'
plt.savefig(s)
print(confusion_matrix)


print('ok')
#'''''