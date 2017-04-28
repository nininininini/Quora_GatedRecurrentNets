#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:00 2017

@author: caradumelvin
"""
import sys
import os
import numpy as np
import time
import nltk.data
import pandas as pd
from tqdm import tqdm
import h5py 
from sklearn.cross_validation import train_test_split
reload(sys)
sys.setdefaultencoding('utf-8')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
'''
Dict_path = '/melvin/Data/Quora/glove.6B.50d.txt'
Data_path = '/melvin/Data/Quora/'
Test_Data_path = '/melvin/Data/Quora/'
'''
Data_path = '/Users/caradumelvin/Documents/Machine Learning - IA/Natural Language Processing/Quora/'
Test_Data_path = Data_path
Dict_path = '/Users/caradumelvin/Documents/Machine Learning - IA/Natural Language Processing/glove.6B/glove.6B.50d.txt'


def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model


#To load tokenizing utilities : nltk.download()
def Split_Tokenize(Data, Training=True):
    X = Data.fillna('empty')
    if Training == True:
        y = X['is_duplicate']
    X1 = []
    X2 = []
    print('Tokenizing data')
    for i in tqdm(range(len(X))):
        X1.append(nltk.wordpunct_tokenize(X.iloc[i, 0]))
        X2.append(nltk.wordpunct_tokenize(X.iloc[i, 1]))
    if Training == True:
        return X1, X2, np.array(y)
    else:
        return X1, X2
    
def Encode_Question(W2V_Dict, Token):
    Q_enc = []
    stop_words = ["a", "able", "about", "across", "after", "all", "almost", "also", "am", "among",
                  "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", 
                  "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every",
                  "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", 
                  "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", 
                  "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor",
                  "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said",
                  "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", 
                  "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", 
                  "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
                  "why", "will", "with", "would", "yet", "you", "your", "ain't", "aren't", "can't", "could've", 
                  "couldn't", "didn't", "doesn't", "don't", "hasn't", "he'd", "he'll", "he's", "how'd", 
                  "how'll", "how's", "i'd", "i'll", "i'm", "i've", "isn't", "it's", "might've", "mightn't",
                  "must've", "mustn't", "shan't", "she'd", "she'll", "she's", "should've", "shouldn't", 
                  "that'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "wasn't", 
                  "we'd", "we'll", "we're", "weren't", "what'd", "what's", "when'd", "when'll", "when's",
                  "where'd", "where'll", "where's", "who'd", "who'll", "who's", "why'd", "why'll", "why's", 
                  "won't", "would've", "wouldn't", "you'd", "you'll", "you're", "you've"]
    
    for word in Token:
        if word not in stop_words:
            try:
                Q_enc.append(W2V_Dict[str(word.lower().encode('utf-8'))])
            except (UnicodeDecodeError, Exception):
                pass
    return sum(Q_enc, [])


def Get_Mask(input_list, max_length):
    mask_i  = np.zeros(shape=(max_length), dtype='float32')
    mask = []
    for q_i in range(len(input_list)):
        mask_i[0:len(input_list[q_i])] = [1] * len(input_list[q_i]) 
        mask.append(mask_i.tolist())
    return np.asarray(mask, 'float32')


def One_Hot_Labels(labels):
    y_enc = np.zeros(shape=(len(labels), 2))
    for l in tqdm(range(len(y_enc))):
        if labels[l] == 1.:
            y_enc[l] = [1., 0.]
        else: 
            y_enc[l] = [0., 1.]
    return y_enc    

def List_to_Arr(list_to_convert, width):
    X_arr = np.zeros(shape=(len(list_to_convert), width))
    for q in range(len(X_arr)):
        X_arr[q,0:len(list_to_convert[q])] = list_to_convert[q]
    X_arr = np.reshape(X_arr, (len(X_arr), width, 1))
    return X_arr
    
def Preprocess_Data(Training_Data, Test_Data, Create_File=False):
    print("Preprocessing Training set:")
    X1, X2, y = Split_Tokenize(Training_Data[['question1','question2', 'is_duplicate']], Training=True)
    y = One_Hot_Labels(np.array(y))
    
    Words_Dict = loadGloveModel(Dict_path)
    
    X = []
    print("Converting words to vectors")
    for question in tqdm(range(len(X1))):
        X.append(Encode_Question(Words_Dict, X1[question]) + 
                 Encode_Question(Words_Dict, X2[question]))
        
    X1, X2= Split_Tokenize(Test_Data[['question1','question2']], Training=False) 
    print("Done")
    print("Preprocessing Test Set:")
    X_test = []
    print("Converting words to vectors")
    for question in tqdm(range(len(X1))):
        X_test.append(Encode_Question(Words_Dict, X1[question]) + 
                 Encode_Question(Words_Dict, X2[question]))
  
    
    global max_len
    max_len = max(max([len(X[i]) for i in range(len(X))]), 
                  max([len(X_test[i]) for i in range(len(X_test))]))
    
    if Create_File == False:
        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                          test_size=0.3, 
                                                          random_state=1260)
        return X_train, X_val, X_test, y_train, y_val
    else:
        #Create file in current directory but fail if file already exists
        f = h5py.File("Quora_Data", "x")
        #Create a group to store the dataset
        grp = f.create_group("X")
        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                          test_size=0.3, 
                                                          random_state=1260)
        
        X_train = List_to_Arr(X_train, max_len)
        #Add training set to the group
        X_train_h5 = grp.create_dataset("X_train", data=X_train, compression="gzip", 
                                compression_opts=9)
        
        y_train_h5 = grp.create_dataset("y_train", data=y_train, compression="gzip", 
                                compression_opts=9)
        del X_train, y_train
        
        #And so on...
        X_val = List_to_Arr(X_val, max_len)
        X_val_h5 = grp.create_dataset("X_val", data=X_val, compression="gzip", 
                                compression_opts=9)

        X_test = List_to_Arr(X_test, max_len)
        X_test_h5 = grp.create_dataset("X_test", data=X_test, compression="gzip", 
                                compression_opts=9)
        y_val_h5 = grp.create_dataset("y_val", data=y_val, compression="gzip", 
                                compression_opts=9)
        del y_val
        
        print("Done")
        
        return f, X_train_h5, X_val_h5, X_test_h5, y_train_h5, y_val_h5


def Get_Data(as_list=True, local=True):
    
    Dataset = pd.read_csv(os.path.join(Data_path, "train.csv"))
    Test_set = pd.read_csv(os.path.join(Test_Data_path, "test.csv"))
    #Reduce the number of samples to fit in an AWS G2.2xlarge memory, e-g:
    #Dataset = Dataset[0:200000]
    #Test_set = Test_set[0:10000]
    
    start_preprocessing = time.time()
    if as_list == True:
        return Preprocess_Data(Dataset, Test_set, Create_File=False)
    else:
        f, X_train_h5, X_val_h5, X_test_h5, y_train_h5, y_val_h5 = Preprocess_Data(Dataset,
                                                                          Test_set,
                                                                          Create_File=True)
        #Get the list of encoded vectors to compute mask for each one
        X_train, X_val, X_test, y_train, y_val = Preprocess_Data(Dataset, 
                                                                 Test_set,
                                                                 Create_File=False)
        
        #Compute mask and add it to the previously created group 
        mask_train = Get_Mask(X_train, max_len)

        grp = f[u'X']
        mask_train_h5 =grp.create_dataset("mask_train", data = mask_train, 
                                          compression="gzip", 
                                          compression_opts=9)
        del X_train, mask_train
        
        mask_test = Get_Mask(X_test, max_len)
        mask_test_h5 = grp.create_dataset("mask_test", data = mask_test,
                                          compression="gzip", 
                                          compression_opts=9)
        del X_test, mask_test
        
        mask_val = Get_Mask(X_val, max_len)
        mask_val_h5 = grp.create_dataset("mask_val", data = mask_val, compression="gzip", 
                                compression_opts=9)
        print('Total preprocessing time: {:.03f} s'.format(time.time()-start_preprocessing))
        return X_train_h5, X_val_h5, X_test_h5, y_train_h5, y_val_h5, mask_train_h5, mask_val_h5, mask_test_h5

