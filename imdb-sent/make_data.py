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
MAX_NB_WORDS = 20000
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


    #########################################################
    lens = [len(review) for review in reviews]
    print('The mean length: {}, Median: {}'.format(np.mean(lens), np.median(lens)))
    #########################################################


    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    print('Tokenizing...')
    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j< MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                        data[i,j,k] = tokenizer.word_index[word]
                        k=k+1                    
                        
    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.seed(0)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Number of positive and negative reviews in traing and validation set') 



    print('loading GLOVE...')
    GLOVE_DIR = "data"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    print('Creating embedding matrix...') 
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('Creating dataset takes {}s.'.format(time.time()-st))
    print('Storing dataset...')  

    np.save('data/x_train.npy', x_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/x_val.npy', x_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/indices.npy', indices)
    np.save('data/embedding_matrix.npy', embedding_matrix)

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
    indices = np.load('data/indices.npy')
    embedding_matrix = np.load('data/embedding_matrix.npy')

    
    dataset = {'x_train': x_train, 'y_train': y_train, 
                'x_val':x_val, 'y_val': y_val, 
                "word_index":word_index, 
                'indices': indices, 
                'embedding_matrix': embedding_matrix}
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

 
  