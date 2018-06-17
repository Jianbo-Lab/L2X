import pandas as pd 
import numpy as np 
import cPickle as pickle
import os 
import csv

def get_selected_words(x_single, score, id_to_word, k): 
	selected_words = {} # {location: word_id}

	selected = np.argsort(score)[-k:] 
	selected_k_hot = np.zeros(400)
	selected_k_hot[selected] = 1.0

	x_selected = (x_single * selected_k_hot).astype(int)
	return x_selected 

def create_dataset_from_score(x, scores, k):
	with open('data/id_to_word.pkl','rb') as f:
		id_to_word = pickle.load(f)
	new_data = []
	new_texts = []
	for i, x_single in enumerate(x):
		x_selected = get_selected_words(x_single, 
			scores[i], id_to_word, k)

		new_data.append(x_selected) 

	np.save('data/x_val-L2X.npy', np.array(new_data))

def calculate_acc(pred, y):
	return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))
	
	


