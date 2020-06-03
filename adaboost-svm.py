#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 00:49:08 2020

@author: sylviaz
"""

import datetime
import random
import numpy as np
from sklearn import svm
#from sklearn.linear_model.logistic import LogisticRegression
import os
#import nltk
#from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


def read_files(subset):
    path = "/Users/sylviaz/Desktop/6010proj/aclImdb/"
    file_list = []
    
    pos_path = path + subset + "/pos/"
    for f in os.listdir(pos_path):
        file_list += [pos_path + f]
        
    neg_path = path + subset + "/neg/"
    for f in os.listdir(neg_path):
        file_list += [neg_path + f]   
        
    all_labels = ([1]*12500+[-1]*12500)
    
    all_texts = []
    for f in file_list:
        with open(f, encoding = 'utf8') as file_input:
            temp = " ".join(file_input.readlines())
            #temp_list = nltk.word_tokenize(temp)
            #filtered = [w for w in temp_list if(w not in stopwords.words('english'))]
            #all_texts += [" ".join(filtered)]
            all_texts += [temp]
            
    return all_labels, all_texts

def error_rate(y,pred):
    return sum(y!= pred)/len(y)

def initclf(X_train,y_train,X_test,y_test,clf):
    clf.fit(X_train,y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_err = error_rate(y_train_pred,y_train)
    test_err = error_rate(y_test_pred,y_test)
    return train_err,test_err

def adaboost(X_train,y_train,X_test,y_test,M,clf):
    w = np.ones(len(X_train))/len(X_train)
    n_train = len(X_train)
    n_test = len(X_test)
    pred_train = [1]*int(n_train/2) + [-1]*(int(n_train)-int(n_train/2))
    pred_test = [1]*int(n_test/2) + [-1]*(int(n_test)-int(n_test/2))
    random.shuffle(pred_train)
    random.shuffle(pred_test)
    for i in range(M):
        print(i, datetime.datetime.now())
        clf.fit(X_train, y_train,sample_weight = w)
        y_train_i = clf.predict(X_train)
        y_test_i = clf.predict(X_test)

        miss = [int(i) for i in (y_train_i != y_train)]
        miss1 = [x if x == 1 else -1 for x in miss]
        error_m =np.dot(w,miss)
        alpha_m = 0.5 *np.log((1-error_m)/error_m)
        z_m = np.dot(w,np.exp([-alpha_m * x for x in miss1 ]))
        w = np.multiply(w,np.exp([-alpha_m * x for x in miss1 ])) / z_m

        pred_train = [sum(x) for x in zip(pred_train,[alpha_m * i for i in y_train_i ])]
        pred_test = [sum(x) for x in zip(pred_test,[alpha_m * i for i in y_test_i ])]
    pred_train,pred_test = np.sign(np.array(pred_train)),np.sign(np.array(pred_test))
    return error_rate(pred_train,y_train), error_rate(pred_test,y_test)

if __name__ =='__main__':
    
    y_train, x_train = read_files("train")
    y_test, x_test = read_files("test")
    
    #y_train = y_train[:2000]+y_train[23000:]
    #y_test = y_test[:1000]+y_test[24000:]
    #x_train = x_train[:2000]+x_train[23000:]
    #x_test = x_test[:1000]+x_test[24000:]
    
    token = Tokenizer(num_words=2000)
    token.fit_on_texts(x_train)
    
    train_seq = token.texts_to_sequences(x_train)
    test_seq = token.texts_to_sequences(x_test)
    
    _train = sequence.pad_sequences(train_seq, maxlen=100)
    _test = sequence.pad_sequences(test_seq, maxlen=100)
    
    clf = svm.SVC()
    er_train,er_test = adaboost(_train,y_train,_test,y_test,30,clf)
