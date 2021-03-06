#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 01:55:56 2020
@author: sylviaz
"""

import os
#import nltk
#from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense,Dropout,Flatten
import pandas as pd
#from keras.layers.recurrent import LSTM
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import AdaBoostClassifier
import sklearn

#get data
def read_files(subset):
    path = "/Users/sylviaz/Desktop/6010proj/aclImdb/"#please change the path before running
    file_list = []
    
    pos_path = path + subset + "/pos/"
    for f in os.listdir(pos_path):
        file_list += [pos_path + f]
        
    neg_path = path + subset + "/neg/"
    for f in os.listdir(neg_path):
        file_list += [neg_path + f]   
            
    all_labels = ([1]*12500+[0]*12500)
    
    all_texts = []
    for f in file_list:
        with open(f, encoding = 'utf8') as file_input:
            temp = " ".join(file_input.readlines())
            #temp_list = nltk.word_tokenize(temp)
            #filtered = [w for w in temp_list if(w not in stopwords.words('english'))]
            #all_texts += [" ".join(filtered)]
            all_texts += [temp]
            
    return all_labels, all_texts

y_train, x_train = read_files("train")
y_test, x_test = read_files("test")

token = Tokenizer(num_words=2000)
token.fit_on_texts(x_train)

train_seq = token.texts_to_sequences(x_train)
test_seq = token.texts_to_sequences(x_test)

_train = sequence.pad_sequences(train_seq, maxlen=100)
_test = sequence.pad_sequences(test_seq, maxlen=100)


#--------------------------------------------------------
#--------------------------------------------------------
model_adaboost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=sklearn.svm.SVC(probability=True))
model_adaboost.fit(_train, y_train)
predict_adaboost = model_adaboost.predict(_test)
predict_adaboost = list(map(lambda x:int(x), predict_adaboost))
pred_pos_num = predict_adaboost.count(1)
pred_neg_sum = predict_adaboost.count(0)
true_pos_num = y_test.count(1)
true_neg_num = y_test.count(0)
right_adaboost = sum(pd.Series(predict_adaboost)==pd.Series(y_test))
right_perc_adaboost = right_adaboost / len(y_test)
adaboost_predpos_truepos = sum((pd.Series(predict_adaboost)==pd.Series(y_test))&(pd.Series(predict_adaboost)==1))
adaboost_predneg_truepos = sum((pd.Series(predict_adaboost)!=pd.Series(y_test))&(pd.Series(predict_adaboost)==0))
adaboost_predpos_trueneg = sum((pd.Series(predict_adaboost)!=pd.Series(y_test))&(pd.Series(predict_adaboost)==1))
adaboost_predneg_trueneg = sum((pd.Series(predict_adaboost)==pd.Series(y_test))&(pd.Series(predict_adaboost)==0))


#--------------------------------------------------------
#--------------------------------------------------------
model_mlp = Sequential()
model_mlp.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
model_mlp.add(Dropout(0.2))

model_mlp.add(Flatten())

model_mlp.add(Dense(units=256, activation='sigmoid'))
model_mlp.add(Dropout(0.2))
model_mlp.add(Dense(units=1, activation='sigmoid'))

model_mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model_mlp.fit(_train, y_train, batch_size=100, epochs=10, verbose=2, validation_split=0.2)

predict_mlp = model_mlp.predict_classes(_test)
predict_mlp = list(map(lambda x:int(x), predict_mlp))
pred_pos_num = predict_mlp.count(1)
pred_neg_sum = predict_mlp.count(0)
#true_pos_num = y_test.count(1)
#true_neg_num = y_test.count(0)
right_mlp = sum(pd.Series(predict_mlp)==pd.Series(y_test))
right_perc_mlp = right_mlp / len(y_test)
mlp_predpos_truepos = sum((pd.Series(predict_mlp)==pd.Series(y_test))&(pd.Series(predict_mlp)==1))
mlp_predneg_truepos = sum((pd.Series(predict_mlp)!=pd.Series(y_test))&(pd.Series(predict_mlp)==0))
mlp_predpos_trueneg = sum((pd.Series(predict_mlp)!=pd.Series(y_test))&(pd.Series(predict_mlp)==1))
mlp_predneg_trueneg = sum((pd.Series(predict_mlp)==pd.Series(y_test))&(pd.Series(predict_mlp)==0))


#--------------------------------------------------------
#--------------------------------------------------------
'''
model_lstm = Sequential()
model_lstm.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
model_lstm.add(Dropout(0.25))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(units=256,activation='relu'))
model_lstm.add(Dropout(0.25))
model_lstm.add(Dense(units=1,activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model_lstm.fit(_train, y_train, batch_size=100, epochs=10, verbose=2, validation_split=0.2)
predict_lstm = model_lstm.predict_classes(_test)
predict_lstm = list(map(lambda x:int(x), predict_lstm))
pred_pos_num = predict_lstm.count(1)
pred_neg_sum = predict_lstm.count(0)
#true_pos_num = y_test.count(1)
#true_neg_num = y_test.count(0)
right_lstm = sum(pd.Series(predict_lstm)==pd.Series(y_test))
right_perc_lstm = right_lstm / len(y_test)
'''


#--------------------------------------------------------
#--------------------------------------------------------
sid = SentimentIntensityAnalyzer()
predict_nltk = []
for sen in x_test:
    ss = sid.polarity_scores(sen)
    score = - 1 * ss["neg"] +1 * ss["pos"]
    y = 0
    if score > 0:
        y = 1
    predict_nltk.append(y)
    
pred_pos_num = predict_nltk.count(1)
pred_neg_sum = predict_nltk.count(0)
#true_pos_num = y_test.count(1)
#true_neg_num = y_test.count(0)
right_nltk = sum(pd.Series(predict_nltk)==pd.Series(y_test))
right_perc_nltk = right_nltk / len(y_test)
nltk_predpos_truepos = sum((pd.Series(predict_nltk)==pd.Series(y_test))&(pd.Series(predict_nltk)==1))
nltk_predneg_truepos = sum((pd.Series(predict_nltk)!=pd.Series(y_test))&(pd.Series(predict_nltk)==0))
nltk_predpos_trueneg = sum((pd.Series(predict_nltk)!=pd.Series(y_test))&(pd.Series(predict_nltk)==1))
nltk_predneg_trueneg = sum((pd.Series(predict_nltk)==pd.Series(y_test))&(pd.Series(predict_nltk)==0))


#--------------------------------------------------------
#--------------------------------------------------------
df = pd.DataFrame()
df["reviews_test"] = pd.Series(x_test)
df["true"] = pd.Series(y_test)
df["adaboost_pred"] = pd.Series(predict_adaboost)
df["mlp_pred"] = pd.Series(predict_mlp)
df["nltk_pred"] = pd.Series(predict_nltk)
df.to_csv("result.csv")