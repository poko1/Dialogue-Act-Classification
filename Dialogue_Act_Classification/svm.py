
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import pickle

#new_tax_train_gf.csv
#new_tax_test_gf.csv
#data/train_da_10.csv data/test_da_10.csv  'clabel', 
df_train = pd.read_csv("/home/maliha/projects/def-zaiane/maliha/data/train_da_10.csv", header =None, sep =',', names = ['text', 'plabel', 'label'], encoding='mac_roman', skiprows=[0]) #utf-8 #'plabel',
df_test = pd.read_csv("/home/maliha/projects/def-zaiane/maliha/data/test_da_10.csv", header = None, sep =',', names = ['text', 'plabel', 'label'], encoding='mac_roman', skiprows=[0]) #'plabel',

df_train = df_train.dropna()
df_test = df_test.dropna()


print('svm')
######################################
#''''
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
#'''''
######################################


X_train = df_train.text
X_test = df_test.text
y_train = df_train.label
y_test = df_test.label


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])

from utils_nlp.common.timer import Timer
with Timer() as t:
    text_clf.fit(X_train, y_train)
train_time = t.interval / 3600

print(train_time)

#save 

#filename = 'svm_model_new_and_old_6.sav'
#pickle.dump(text_clf, open(filename, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#predicted = loaded_model.predict(X_test)
#print(metrics.classification_report(y_test, predicted))


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))

import matplotlib.pyplot as plt
import numpy as np
confusion_matrix = metrics.confusion_matrix(y_test, predicted)
#confusion_matrix = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=1)
confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
#'A','DD','QF','G','DI','F','S','QYN'    'I', 'Q', 'S'
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['A','DD','QF','G','DI','F','S','QYN']) #['D_D','Q_F','S_IS','S_NIS','D_R','Q_YN']  ['D', 'Q', 'S'] ['D_D','Q_F','S_S','D_I','Q_YN']
c = cm_display.plot()
s = 'svm_8.jpg'
plt.savefig(s)
print(confusion_matrix)

# print index of wrongly predicted
print("'''''''")
n = ['g',] #, 'n', 'p', 'c'
l = 0
for idx, input in df_test.iterrows(): 
    if predicted[idx] != y_test[idx]:
        l = l+1
        if y_test[idx] in n: # or predicted[idx] in n: 
            print("No.", idx, 'input,',input.text, ', has been classified as', predicted[idx], 'and should be', y_test[idx]) 

print(l)

