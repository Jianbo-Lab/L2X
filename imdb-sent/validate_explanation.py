"""
Compute the accuracy with selected sentences by using L2X.
"""
from __future__ import print_function 
import numpy as np
import pandas as pd
try:
   import cPickle as pkl
except:
   import pickle as pkl

from collections import defaultdict
import re 
from bs4 import BeautifulSoup 
import sys
import os
import time  
from make_data import load_data 
import pandas as pd 
import json
import tensorflow as tf
import csv
from explain import create_original_model

def validate(): 
	print('Loading dataset...') 
	dataset = load_data()
	word_index = dataset['word_index']  
	x_val = np.load('data/x_val-L2X.npy') 
	pred_val = np.load('data/pred_val.npy') 
				
	print('Creating model...')
	model = create_original_model(word_index)
	model.load_weights('./models/original.hdf5', 
		by_name=True) 

	print('Making prediction with selected sentences...')
	new_pred_val = model.predict(x_val, verbose = 1, batch_size = 1000)
	val_acc = np.mean(np.argmax(new_pred_val, axis = -1)==np.argmax(pred_val, axis = -1))

	print('the validation accuracy is {}.'.format(val_acc)) 
	np.save('data/pred_val-{}.npy'.format('L2X'), new_pred_val)

if __name__ == '__main__': 
	validate()








