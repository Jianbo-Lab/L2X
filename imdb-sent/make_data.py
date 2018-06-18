"""
Modified based on the code of Richard Liao at https://github.com/richliao/textClassifier
"""
from __future__ import print_function 
import numpy as np 
try:
   import cPickle as pkl
except:
   import pickle as pkl

from collections import defaultdict
import re
from nltk import tokenize
from bs4 import BeautifulSoup
import pandas as pd
import sys
import os
import time 
import json 
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset.
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


def create_dataset():
    """
    Create the IMDB dataset as numpy arrays.
    
    """
    st = time.time()
    print('Constructing dataset...')
    data_train = pd.read_csv('data/labeledTrainData.tsv', sep='\t') 
    data_test = pd.read_csv('data/testData.tsv', sep='\t')

    from nltk import tokenize

    reviews = []
    labels = []
    texts = []

    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        text = clean_str(text.get_text().encode('ascii','ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        
        labels.append(data_train.sentiment[idx])


    for idx in range(data_test.review.shape[0]):
        text = BeautifulSoup(data_test.review[idx])
        text = clean_str(text.get_text().encode('ascii','ignore'))
        texts.append(text) # texts is the raw text
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

        if data_test.id[idx][-1] in "12345":
            labels.append(0)
        else:
            labels.append(1)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)

    print('Tokenizing...')
    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j< MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<=MAX_NUM_WORDS:
                        data[i,j,k] = tokenizer.word_index[word]
                        k=k+1                    
                        
    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    x_train = data[:25000]
    y_train = labels[:25000]
    x_val = data[25000:]
    y_val = labels[25000:]

    print('Number of positive and negative reviews in traing and validation set') 

    print('Creating dataset takes {}s.'.format(time.time()-st))
    print('Storing dataset...')  

    np.save('data/x_train.npy', x_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/x_val.npy', x_val)
    np.save('data/y_val.npy', y_val) 

    with open('data/word_index.pkl','wb') as f:
        pkl.dump(word_index, f) 

def load_data():  
    """
    Load data if data have been created.
    Create data otherwise.
    
    """

    if 'data' not in os.listdir('.'):
        os.mkdir('data')
    if 'word_index.pkl' not in os.listdir('data'): 
        create_dataset()


    with open('./data/word_index.pkl','rb') as f:
        word_index = pkl.load(f) 
    x_train = np.load('data/x_train.npy')
    y_train = np.load('data/y_train.npy')
    x_val = np.load('data/x_val.npy')
    y_val = np.load('data/y_val.npy')

    dataset = {'x_train': x_train, 'y_train': y_train, 
                'x_val':x_val, 'y_val': y_val, 
                "word_index":word_index
                }
    print('Data loaded...') 
    return dataset

def create_dataset_from_score(scores, x):
    """
    Construct data set containing selected sentences by L2X.

    """
    if len(scores.shape) == 3:
        scores = np.squeeze(scores) 
    sent_ids = np.argmax(scores, axis = 1)

    x_new = np.zeros(x.shape)
    for i, sent_id in enumerate(sent_ids):
        x_new[i,sent_id,:] = x[i][sent_id]

    np.save('data/x_val-L2X.npy',np.array(x_new))

 
  