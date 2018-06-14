"""
References:
The code of Richard Liao at https://github.com/richliao/textClassifier

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
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda, Permute
from keras.layers import Activation, Conv1D, GlobalMaxPooling1D,MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Reshape
from keras.models import Model
import argparse
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from make_data import load_data 
import pandas as pd 
import json
import tensorflow as tf
import csv
tf.set_random_seed(10086)
np.random.seed(10086)
MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
BATCHSIZE = 100
###################################
##########Original Model###########
###################################

def create_original_model(embedding_matrix, word_index):
	with tf.variable_scope('prediction_model'): 
		embedding_layer = Embedding(len(word_index) + 1,
										EMBEDDING_DIM,
										weights=[embedding_matrix],
										input_length=MAX_SENT_LENGTH,
										name = 'embedding',
										trainable=True)

		sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
		embedded_sequences = embedding_layer(sentence_input)
		l_lstm = Bidirectional(LSTM(100, name = 'lstm'), 
			name = 'bidirectional')(embedded_sequences)
		sentEncoder = Model(sentence_input, l_lstm)

		review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
		review_encoder = TimeDistributed(sentEncoder)(review_input)
		l_lstm_sent = Bidirectional(LSTM(100, name = 'lstm2'), 
			name = 'bidirectional2')(review_encoder)
		preds = Dense(2, activation='softmax', name = 'dense')(l_lstm_sent)
		model = Model(review_input, preds)

		model.compile(loss='categorical_crossentropy',
					  optimizer='rmsprop',
					  metrics=['acc'])
		return model 	

def original(train = True): 
	print('Loading data...') 
	dataset = load_data()
	word_index = dataset['word_index']
	x_train, x_val, y_train, y_val = dataset['x_train'], \
	dataset['x_val'], dataset['y_train'], dataset['y_val']
	indices = dataset['indices'] 
	embedding_matrix = dataset['embedding_matrix']
				
	print('Creating model...')
	model = create_original_model(embedding_matrix, word_index)

	if train:
		if 'models' not in os.listdir('.'):
			os.mkdir('models')

		filepath="models/original.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, 
			epochs=10, batch_size=BATCHSIZE)

	else:
		weights_name = 'original.hdf5'
		model.load_weights('./models/' + weights_name, 
			by_name=True) 

	pred_train = model.predict(x_train,verbose = 1, batch_size = 1000)
	pred_val = model.predict(x_val,verbose = 1, batch_size = 1000)

	np.save('data/pred_train.npy', pred_train)
	np.save('data/pred_val.npy', pred_val)



###################################
##########     L2X      ###########
###################################

k = 1 
Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k), 
	output_shape=lambda x: [x[0],x[2]]) 

from keras.engine.topology import Layer

class Sample_Concrete(Layer):

	def __init__(self, tau0, k,**kwargs): 
		self.tau0 = tau0
		self.k = k
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):
		logits_ = K.permute_dimensions(logits, (0,2,1))# [batchsize, 1, MAX_SENTS]

		unif_shape = tf.shape(logits_)[0]
		uniform = tf.random_uniform(shape =(unif_shape, self.k, MAX_SENTS), 
			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
			maxval = 1.0)

		gumbel = - K.log(-K.log(uniform))
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = K.softmax(noisy_logits)
		samples = K.max(samples, axis = 1)
		samples = K.expand_dims(samples, -1)

		discrete_logits = K.one_hot(K.argmax(logits_,axis=-1), num_classes = MAX_SENTS)
		discrete_logits = K.permute_dimensions(discrete_logits, 
			(0,2,1))
		return K.in_train_phase(samples, discrete_logits)

	def compute_output_shape(self, input_shape):
		return input_shape


class Concatenate(Layer):
	def __init__(self, **kwargs): 
		super(Concatenate, self).__init__(**kwargs)

	def call(self, inputs):
		input1, input2 = inputs  
		input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
		dim1 = int(input2.get_shape()[1])
		input1 = tf.tile(input1, [1, dim1, 1])
		return tf.concat([input1, input2], axis = -1)

	def compute_output_shape(self, input_shapes):
		input_shape1, input_shape2 = input_shapes
		input_shape = list(input_shape2)
		input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
		input_shape[-2] = int(input_shape[-2])
		return tuple(input_shape)

def create_dataset_from_score(scores, x):
	if len(scores.shape) == 3:
		scores = np.squeeze(scores) 
	sent_ids = np.argmax(scores, axis = 1)

	# added. 
	x_new = np.zeros(x.shape)
	for i, sent_id in enumerate(sent_ids):
		x_new[i,sent_id,:] = x[i][sent_id]

	np.save('data/x_val-L2X.npy',np.array(x_new))

def L2X(train = True): 
	print('Loading data...') 
	dataset = load_data()
	word_index = dataset['word_index']
	x_train, x_val, y_train, y_val = dataset['x_train'], dataset['x_val'], dataset['y_train'], dataset['y_val']
	indices = dataset['indices'] 
	embedding_matrix = dataset['embedding_matrix']
	with open('./data/word_index.pkl','rb') as f:
		word_index = pkl.load(f) 

	print('Creating model...')

	# P(S|X)
	with tf.variable_scope('selection_model'):
		sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
		embedding_layer = Embedding(len(word_index) + 1,
									EMBEDDING_DIM,
									weights=[embedding_matrix],
									input_length=MAX_SENT_LENGTH,
									name = 'embedding',
									trainable=True)

		embedded_sequences = embedding_layer(sentence_input)
		net = Dropout(0.2)(embedded_sequences)
		net = Conv1D(250,
					 3,
					 padding='valid',
					 activation='relu',
					 strides=1)(net)
		net = GlobalMaxPooling1D()(net)
		sentEncoder = Model(sentence_input, net) 

		review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')

		review_encoder = TimeDistributed(sentEncoder)(review_input) # [batch_size, max_sents, 100] 
		tau = 0.5 
		kernel_size = 3
		net = review_encoder
		first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(net)  

		filters = 100; kernel_size = 3; hidden_dims = 100  

		# global info
		# we use max pooling:
		net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
		# We add a vanilla hidden layer:
		global_info = Dense(hidden_dims, name = 'new_dense_1', activation='relu')(net_new) 

		# local info
		net = Conv1D(50, kernel_size, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer) 
		local_info = Conv1D(50, kernel_size, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net)  
		combined = Concatenate()([global_info,local_info]) 
		net = Dropout(0.2, name = 'new_dropout_2')(combined)
		net = Conv1D(50, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)   

		logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)  

		T = Sample_Concrete(tau, k)(logits_T)

	# q(X_S)
	with tf.variable_scope('prediction_model'):  
		sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
		embedding_layer = Embedding(len(word_index) + 1,
									EMBEDDING_DIM,
									weights=[embedding_matrix],
									input_length=MAX_SENT_LENGTH,
									name = 'embedding',
									trainable=True)

		embedded_sequences = embedding_layer(sentence_input)
		net = Dropout(0.2)(embedded_sequences)
		net = Conv1D(250,
					 3,
					 padding='valid',
					 activation='relu',
					 strides=1)(net)
		net = GlobalMaxPooling1D()(net)
		sentEncoder2 = Model(sentence_input, net) 

		review_encoder2 = TimeDistributed(sentEncoder2)(review_input)
		selected_encoding = Multiply()([review_encoder2, T])
		net = Mean(selected_encoding)
		net = Dense(250)(net)
		# net = Dropout(0.2)(net)
		net = Activation('relu')(net) 
		preds = Dense(2, activation='softmax', 
			name = 'new_dense')(net)

	model = Model(inputs=review_input, 
		outputs=preds)

	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['acc'])  

	pred_train = np.load('data/pred_train.npy')  
	pred_val = np.load('data/pred_val.npy')  

	val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
	print('The train and validation accuracy of the original model is {}'.format(val_acc))

	if train:
		filepath="models/l2x.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint] 

		

		model.fit(x_train, pred_train, 
			validation_data=(x_val, pred_val),
			callbacks = callbacks_list,
			epochs=10, batch_size=BATCHSIZE)

	else:
		weights_name = 'l2x.hdf5'
		model.load_weights('models/{}'.format(weights_name), 
			by_name=True) 


	pred_model = Model(review_input, [T,logits_T,preds])
	
	pred_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

	st = time.time()
	selections, scores, interp_val = pred_model.predict(x_val, 
		verbose = 1, 
		batch_size = BATCHSIZE) 

	print('Creating dataset with selected sentences...')
	create_dataset_from_score(scores, x_val)
	print('Time spent is {}'.format(time.time() - st)) 


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser() 
	parser.add_argument('--task', type = str, choices = ['original','L2X'], default = 'L2X') 
	parser.add_argument('--train', action='store_true') 
	parser.set_defaults(train=False)

	args = parser.parse_args()
	dict_a = vars(args)   
	if args.task == 'L2X':
		L2X(args.train)
	elif args.task == 'original':
		original(args.train)











