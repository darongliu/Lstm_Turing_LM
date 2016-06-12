import sys
import os.path
#sys.path.insert(0, '/home/andy/theano/tool_examples/theano-lstm-0.0.15')
from theano_lstm import Embedding, LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss
from utilities import *

import dill
import argparse
#import cPickle
import pickle
import numpy
from collections import OrderedDict
import theano, theano.tensor as T
import turing_model
from theano_toolkit.parameters import Parameters
from theano.compile.nanguardmode import NanGuardMode

DESCRIPTION = """
	Recurrent neural network based statistical language modelling toolkit
	(based on LSTM algorithm)
	Implemented by Daniel Soutner,
	Department of Cybernetics, University of West Bohemia, Plzen, Czech rep.
	dsoutner@kky.zcu.cz, 2013
"""

def parse_args(parser):
	parser.add_argument('--train', nargs=1, action="store", metavar="FILE",
						help='training file !')

	parser.add_argument('--valid', nargs=1, action="store", metavar="FILE",
						help='valid file !')

	parser.add_argument('--test', nargs=1, action="store", metavar="FILE",
						help='testing file for ppl!')

	parser.add_argument('--neuron-type', action="store", dest='celltype',
						help='type of hidden neurons, RNN/LSTM, default: RNN', type=str, default='RNN')

	parser.add_argument('--train-method', action="store", dest='train_method',
						help='training method LSTM/TURING/ALL, default: ALL', type=str, default='ALL')

	parser.add_argument('--projection-size', action="store", dest='n_projection',
						help='Number of neurons in projection layer, default: 100', type=int, default=100)

	parser.add_argument('--hidden-size', action="store", dest='n_hidden',
						help='Number of neurons in hidden layer, default: 100', type=int, default=100)

	parser.add_argument('--stack', action="store", dest='n_stack',
						help='Number of hidden neurons, default: 1 ', type=int, default=1)

	parser.add_argument('--learning-rate', action="store", dest='lr',
						help='learing rate at begining, default: 0.01 ', type=float, default=0.01)

	parser.add_argument('--improvement-rate', action="store", dest='improvement_rate',
						help='relative improvement for early stopping on ppl , default: 0.005 ', type=float, default=0.005)

	parser.add_argument('--minibatch-size', action="store", dest='minibatch_size',
						help='minibatch size for training, default: 100', type=int, default=100)

	parser.add_argument('--max-epoch', action="store", dest='max_epoch',
						help='maximum number of epoch if not early stopping, default: 1000', type=int, default=1000)

	parser.add_argument('--early-stop', action="store", dest='early_stop',
						help='1 for early-stopping, 0 for not', type=int, default=1)

	parser.add_argument('--save-net', action="store", dest="save_net", default=None, metavar="FILE",
						help="Save RNN to file")

	parser.add_argument('--load-net', action="store", dest="load_net", default=None, metavar="FILE",
						help="Load RNN from file")

	return parser.parse_args()


def build_vocab(data_file_str):

	lines = []
	data_file = open(data_file_str)
	for line in data_file:
		tokens = line.replace('\n','.')
		lines.append(tokens)
	data_file.close()

	vocab = Vocab()
	for line in lines:
		vocab.add_words(line.split(" "))

	return vocab

def load_data(data_file_str, vocab, data_type):

	lines = []
	data_file = open(data_file_str)
	for line in data_file:
		tokens = line.replace('\n','.')

		# abandom too long sent in training set., too long sent will take too many time and decrease preformance
		tokens_for_count = line.replace('\n','').split(' ')
		if len(tokens_for_count) > 50 and data_type == 'train':
			continue

		lines.append(tokens)
	data_file.close()

	# transform into big numerical matrix of sentences:
	numerical_lines = []
	for line in lines:
		numerical_lines.append(vocab(line))
	numerical_lines, numerical_lengths = pad_into_matrix(numerical_lines)

	return numerical_lines, numerical_lengths


def softmax(x):
	"""
	Wrapper for softmax, helps with
	pickling, and removing one extra
	dimension that Theano adds during
	its exponential normalization.
	"""
	return T.nnet.softmax(x.T)

def has_hidden(layer):
	"""
	Whether a layer has a trainable
	initial hidden state.
	"""
	return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
	return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
	"""
	Initalizes the recurrence relation with an initial hidden state
	if needed, else replaces with a "None" to tell Theano that
	the network **will** return something, but it does not need
	to send it to the next step of the recurrence
	"""
	if dimensions is None:
		return layer.initial_hidden_state if has_hidden(layer) else None
	else:
		return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None
	
def initial_state_with_taps(layer, dimensions = None):
	"""Optionally wrap tensor variable into a dict with taps=[-1]"""
	state = initial_state(layer, dimensions)
	if state is not None:
		return dict(initial=state, taps=[-1])
	else:
		return None

class Model:
	"""
	Simple predictive model for forecasting words from
	sequence using LSTMs. Choose how many LSTMs to stack
	what size their memory should be, and how many
	words can be predicted.
	"""
	def __init__(self, hidden_size, input_size, vocab_size, stack_size=1, celltype=LSTM):

		# core layer in RNN/LSTM
		self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)

		# add an embedding
		self.model.layers.insert(0, Embedding(vocab_size, input_size))

		# add a classifier:
		self.model.layers.append(Layer(hidden_size, vocab_size, activation = softmax))

                self.turing_params = Parameters()
		#init turing machine model
		self.turing_updates , self.turing_predict = turing_model.build(self.turing_params , hidden_size , vocab_size)

		# inputs are matrices of indices,
		# each row is a sentence, each column a timestep
		self._stop_word   = theano.shared(np.int32(999999999), name="stop word")
		self.for_how_long = T.ivector()
		self.input_mat = T.imatrix()
		self.priming_word = T.iscalar()
		self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

		# create symbolic variables for prediction:
		#change by darong #issue : what is greedy
		self.lstm_predictions = self.create_lstm_prediction()
		self.final_predictions = self.create_final_prediction()

		# create symbolic variable for greedy search:
		self.greedy_predictions = self.create_lstm_prediction(greedy=True)

		# create gradient training functions:
		self.create_cost_fun()#create 2 cost func(lstm final)

		self.lstm_lr = 0.01
		self.turing_lr = 0.01
		self.all_lr = 0.01
		self.create_training_function()#create 3 functions(lstm turing all)
		self.create_predict_function()#create 2 predictions(lstm final)

		# create ppl
		self.lstm_ppl = self.create_lstm_ppl()
		self.final_ppl = self.create_final_ppl()
		self.create_ppl_function()


	def save(self, save_file, vocab):
		pickle.dump(self.model, open(save_file, "wb")) # pickle is for lambda function, cPickle cannot
		pickle.dump(vocab, open(save_file+'.vocab', "wb")) # pickle is for lambda function, cPickle cannot
	def save_turing(self, save_file):
		self.turing_params.save(save_file + '.turing')


	def load(self, load_file, lr):
		self.model = pickle.load(open(load_file, "rb"))
		if os.path.isfile(load_file + '.turing') :
			self.turing_params.load(load_file + '.turing')			
		else :
			print "no turing model!!!! pretrain with lstm param"
			self.turing_params['W_input_hidden'] = self.model.layers[-1].params[0].get_value().T #not sure
			self.turing_params['W_read_hidden']  = self.model.layers[-1].params[0].get_value().T
			self.turing_params['b_hidden_0'] = self.model.layers[-1].params[1].get_value()

		# need to compile again for calculating predictions after loading lstm
		self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))
		self.lstm_predictions = self.create_lstm_prediction()
		self.final_predictions = self.create_final_prediction()
		self.greedy_predictions = self.create_lstm_prediction(greedy=True)#can change to final
		self.create_cost_fun()#create 2 cost func(lstm final)
		self.lstm_lr = lr
		self.turing_lr = lr#change this
		self.all_lr = lr
		self.create_training_function()#create 3 functions(lstm turing all)
		self.create_predict_function()#create 2 predictions(lstm final)
		self.lstm_ppl = self.create_lstm_ppl()
		self.final_ppl = self.create_final_ppl()
		self.create_ppl_function()
		print "done loading model"
#		print "done compile"


	def stop_on(self, idx):
		self._stop_word.set_value(idx)
		
	@property
	def params(self):
		return self.model.params
								 
	def create_lstm_prediction(self, greedy=False):
		def step(idx, *states):
			# new hiddens are the states we need to pass to LSTMs
			# from past. Because the StackedCells also include
			# the embeddings, and those have no state, we pass
			# a "None" instead:
			new_hiddens = [None] + list(states)
			
			new_states = self.model.forward(idx, prev_hiddens = new_hiddens)
			if greedy:
				new_idxes = new_states[-1]
				new_idx   = new_idxes.argmax()
				# provide a stopping condition for greedy search:
				return ([new_idx.astype(self.priming_word.dtype)] + new_states[1:-1]), theano.scan_module.until(T.eq(new_idx,self._stop_word))
			else:
				return new_states[1:]

		# in sequence forecasting scenario we take everything
		# up to the before last step, and predict subsequent
		# steps ergo, 0 ... n - 1, hence:
		inputs = self.input_mat[:, 0:-1]
		num_examples = inputs.shape[0]
		# pass this to Theano's recurrence relation function:
		
		# choose what gets outputted at each timestep:
		if greedy:
			outputs_info = [dict(initial=self.priming_word, taps=[-1])] + [initial_state_with_taps(layer) for layer in self.model.layers[1:-1]]
			result, _ = theano.scan(fn=step,
								n_steps=200,
								outputs_info=outputs_info)
		else:
			outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers[1:]]
			result, _ = theano.scan(fn=step,
								sequences=[inputs.T],
								outputs_info=outputs_info)
								 
		if greedy:
			return result[0]
		# softmaxes are the last layer of our network,
		# and are at the end of our results list:
		return result[-1].transpose((2,0,1))
		# we reorder the predictions to be:
		# 1. what row / example
		# 2. what timestep
		# 3. softmax dimension

	def create_final_prediction(self, greedy=False):
		def step(idx, *states):
			# new hiddens are the states we need to pass to LSTMs
			# from past. Because the StackedCells also include
			# the embeddings, and those have no state, we pass
			# a "None" instead:
			new_hiddens = [None] + list(states)
			
			new_states = self.model.forward(idx, prev_hiddens = new_hiddens)
			if greedy:
				new_idxes = new_states[-1]
				new_idx   = new_idxes.argmax()
				# provide a stopping condition for greedy search:
				return ([new_idx.astype(self.priming_word.dtype)] + new_states[1:-1]), theano.scan_module.until(T.eq(new_idx,self._stop_word))
			else:
				return new_states[1:]

		# in sequence forecasting scenario we take everything
		# up to the before last step, and predict subsequent
		# steps ergo, 0 ... n - 1, hence:
		inputs = self.input_mat[:, 0:-1]
		num_examples = inputs.shape[0]
		# pass this to Theano's recurrence relation function:
		
		# choose what gets outputted at each timestep:
		if greedy:
			outputs_info = [dict(initial=self.priming_word, taps=[-1])] + [initial_state_with_taps(layer) for layer in self.model.layers[1:-1]]
			result, _ = theano.scan(fn=step,
								n_steps=200,
								outputs_info=outputs_info)
		else:
			outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers[1:]]
			result, _ = theano.scan(fn=step,
								sequences=[inputs.T],
								outputs_info=outputs_info)
								 
		if greedy:
			return result[0]
		# softmaxes are the last layer of our network,
		# and are at the end of our results list:
                hidden_size = result[-2].shape[2]/2
		turing_result = self.turing_predict(result[-2][:,:,hidden_size:]) 
		#the last layer do transpose before compute
		return turing_result.transpose((1,0,2))
		# we reorder the predictions to be:
		# 1. what row / example
		# 2. what timestep
		# 3. softmax dimension	
								 
	def create_cost_fun (self):

		# create a cost function that
		# takes each prediction at every timestep
		# and guesses next timestep's value:
		what_to_predict = self.input_mat[:, 1:]
		# because some sentences are shorter, we
		# place masks where the sentences end:
		# (for how long is zero indexed, e.g. an example going from `[2,3)`)
		# has this value set 0 (here we substract by 1):
		for_how_long = self.for_how_long - 1
		# all sentences start at T=0:
		starting_when = T.zeros_like(self.for_how_long)
								 
		self.lstm_cost = masked_loss(self.lstm_predictions,
								what_to_predict,
								for_how_long,
								starting_when).sum()

		self.final_cost = masked_loss(self.final_predictions,
								what_to_predict,
								for_how_long,
								starting_when).sum()
		
	def create_predict_function(self):
		self.lstm_pred_fun = theano.function(
			inputs=[self.input_mat],
			outputs=self.lstm_predictions,
			allow_input_downcast=True
		)
		self.final_pred_fun = theano.function(
			inputs=[self.input_mat],
			outputs=self.final_predictions,
			allow_input_downcast=True
		)
		
		self.greedy_fun = theano.function(
			inputs=[self.priming_word],
			outputs=T.concatenate([T.shape_padleft(self.priming_word), self.greedy_predictions]),
			allow_input_downcast=True
		)
								 
	def create_training_function(self):
		updates, _, _, _, _ = create_optimization_updates(self.lstm_cost, self.params, method="SGD", lr=self.lstm_lr)
#		updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta", lr=self.lr)
		self.lstm_update_fun = theano.function(
			inputs=[self.input_mat, self.for_how_long],
			outputs=self.lstm_cost,
			updates=updates,
			allow_input_downcast=True)

		updates_turing = self.turing_updates(self.final_cost , lr=self.turing_lr)
#		updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta", lr=self.lr)
		self.turing_update_fun = theano.function(
			inputs=[self.input_mat, self.for_how_long],
			outputs=self.final_cost,
			updates=updates_turing,
                        mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
			allow_input_downcast=True)

		all_updates_lstm, _, _, _, _ = create_optimization_updates(self.final_cost, self.params, method="SGD", lr=self.all_lr,part=True)
		all_updates_turing_temp = self.turing_updates(self.final_cost , lr=self.all_lr)
                updates_all = all_updates_lstm
                for pair in all_updates_turing_temp :
                    updates_all[pair[0]] = pair[1]

		self.all_update_fun = theano.function(
			inputs=[self.input_mat, self.for_how_long],
			outputs=self.final_cost,
			updates=updates_all,
			allow_input_downcast=True)

	def create_lstm_ppl(self):

		def timestep(predictions, label, len_example, total_len_example):

			label_binary = T.gt(label[0:len_example-1], 0)
			oov_count = T.shape(label_binary)[0] - T.sum(label_binary)
			
			a = total_len_example
			return T.sum(T.log( 1./ predictions[T.arange(len_example-1), label[0:len_example-1]]) * label_binary ), oov_count


		result, _ = theano.scan(fn=timestep,
						   sequences=[ self.lstm_predictions, self.input_mat[:, 1:], self.for_how_long ],
						   non_sequences=T.sum(self.for_how_long))

		oov_count_total = T.sum(result[1])
		return T.exp(T.sum(result[0]).astype(theano.config.floatX)/(T.sum(self.for_how_long) - oov_count_total).astype(theano.config.floatX)).astype(theano.config.floatX)

	def create_final_ppl(self):

		def timestep(predictions, label, len_example, total_len_example):

			label_binary = T.gt(label[0:len_example-1], 0)
			oov_count = T.shape(label_binary)[0] - T.sum(label_binary)
			
			a = total_len_example
			return T.sum(T.log( 1./ predictions[T.arange(len_example-1), label[0:len_example-1]]) * label_binary ), oov_count


		result, _ = theano.scan(fn=timestep,
						   sequences=[ self.final_predictions, self.input_mat[:, 1:], self.for_how_long ],
						   non_sequences=T.sum(self.for_how_long))

		oov_count_total = T.sum(result[1])
		return T.exp(T.sum(result[0]).astype(theano.config.floatX)/(T.sum(self.for_how_long) - oov_count_total).astype(theano.config.floatX)).astype(theano.config.floatX)

	def create_ppl_function(self):
		self.lstm_ppl_fun = theano.function(
			inputs=[self.input_mat, self.for_how_long],
			outputs=self.lstm_ppl,
			allow_input_downcast=True)

		self.final_ppl_fun = theano.function(
			inputs=[self.input_mat, self.for_how_long],
			outputs=self.final_ppl,
			allow_input_downcast=True)
		
		
	def __call__(self, x):
		return self.pred_fun(x)#any problem??

def get_minibatch(full_data, full_lengths, minibatch_size, minibatch_idx):
	lengths = []
	for j in range(minibatch_size):
		lengths.append(full_lengths[minibatch_size * minibatch_idx + j])

	width = max(full_lengths)
#	width = max(full_data[minibatch_size * minibatch_idx: minibatch_size * (minibatch_idx+1), :])
	height = minibatch_size
	minibatch_data = np.empty([height, width], dtype=theano.config.floatX)
	minibatch_data = full_data[minibatch_size * minibatch_idx: minibatch_size * (minibatch_idx+1), :]

	return minibatch_data, lengths

def training(args, vocab, train_data, train_lengths, valid_data, valid_lengths):

	# training information
	print 'training information'
	print '-------------------------------------------------------'
	print 'method: %s' % args.train_method
	print 'vocab size: %d' % len(vocab)
	print 'sentences in training file: %d' % len(train_lengths)
	print 'max length in training file: %d' % max(train_lengths) 
	print 'train file: %s' % args.train[0]
	print 'valid file: %s' % args.valid[0]
	print 'type: %s' % args.celltype
	print 'project: %d' % args.n_projection
	print 'hidden: %d' % args.n_hidden
	print 'stack: %d' % args.n_stack
	print 'learning rate: %f' % args.lr
	print 'minibatch size: %d' % args.minibatch_size
	print 'max epoch: %d' % args.max_epoch
	print 'improvement rate: %f' % args.improvement_rate
	print 'save file: %s' % args.save_net
	print 'load_model: %s' % args.load_net
	print 'early-stop: %r' % args.early_stop
	print '-------------------------------------------------------'

	if args.celltype == 'LSTM':
		celltype = LSTM
	elif args.celltype == 'RNN':
		celltype = RNN

	print 'start initializing model'
	# construct model & theano functions:
	model = Model(
		input_size=args.n_projection,
		hidden_size=args.n_hidden,
		vocab_size=len(vocab),
		stack_size=args.n_stack, # make this bigger, but makes compilation slow
		celltype=celltype # use RNN or LSTM
	)
        if args.lr :
                model.lstm_lr   = args.lr
                model.turing_lr = args.lr
                model.all_lr    = args.lr

	model.stop_on(vocab.word2index["."])
        """
	if args.load_net :
                if args.lr :
		        model.load(args.load_net, args.lr)# 0 is useless
                else :
                        model.load(args.load_net, 0)
	"""
	# train:
	#select correct train and prediction method according to train_method(LSTM/TURING/ALL)
	if args.train_method == 'LSTM' :
		update_fun = model.lstm_update_fun
		ppl_fun    = model.lstm_ppl_fun
		lr         = model.lstm_lr
                print 'update lstm learning rate : %f' % model.lstm_lr
	elif args.train_method == 'TURING' :
		update_fun = model.turing_update_fun
		ppl_fun    = model.final_ppl_fun
		lr         = model.turing_lr
                print 'update turing learning rate : %f' % model.turing_lr
	else :
		update_fun = model.all_update_fun
		ppl_fun    = model.final_ppl_fun
		lr         = model.all_lr
                print 'update all learning rate : %f' % model.all_lr

	stop_count = 0 # for stop training
	change_count = 0 # for change learning rate
	print 'start training'
	min_valid_ppl = float('inf')
	for epoch in range(args.max_epoch):
		print "\nepoch %d" % epoch

		# minibatch part
		minibatch_size = args.minibatch_size # how many examples in a minibatch
		n_train_batches = len(train_lengths)/minibatch_size

		train_ppl = 0
		for minibatch_idx in range(n_train_batches):
			minibatch_train_data, lengths = get_minibatch(train_data, train_lengths, minibatch_size, minibatch_idx)
			error = update_fun(minibatch_train_data , list(lengths) )
			minibatch_train_ppl = ppl_fun(minibatch_train_data, list(lengths))
			train_ppl = train_ppl + minibatch_train_ppl * sum(lengths)
			sys.stdout.write( '\n%d minibatch idx / %d total minibatch, ppl: %f '% (minibatch_idx+1, n_train_batches, minibatch_train_ppl) )
			sys.stdout.flush() # important

		# rest minibatch if exits
		if (minibatch_idx + 1) * minibatch_size != len(train_lengths):
			minibatch_idx = minibatch_idx + 1
			n_rest_example = len(train_lengths) - minibatch_size * minibatch_idx
			minibatch_train_data, lengths = get_minibatch(train_data, train_lengths, n_rest_example, minibatch_idx)
	
			error = update_fun(minibatch_train_data , list(lengths) )
			minibatch_train_ppl = ppl_fun(minibatch_train_data, list(lengths))
			train_ppl = train_ppl + minibatch_train_ppl * sum(lengths)

		train_ppl = train_ppl / sum(train_lengths)
#		print 'done training'

		# valid ppl
		minibatch_size = min(20, len(valid_lengths))
		valid_ppl = 0
		n_valid_batches = len(valid_lengths)/minibatch_size
		for minibatch_idx in range(n_valid_batches):
			minibatch_valid_data, lengths = get_minibatch(valid_data, valid_lengths, minibatch_size, minibatch_idx)
			minibatch_valid_ppl = ppl_fun(minibatch_valid_data, list(lengths))
			valid_ppl = valid_ppl + minibatch_valid_ppl * sum(lengths)
			
		# last minibatch
		if (minibatch_idx + 1) * minibatch_size != len(valid_lengths):
			minibatch_idx = minibatch_idx + 1
			n_rest_example = len(valid_lengths) - minibatch_size * minibatch_idx
			minibatch_valid_data, lengths = get_minibatch(valid_data, valid_lengths, n_rest_example, minibatch_idx)

			minibatch_valid_ppl = ppl_fun(minibatch_valid_data, list(lengths))
			valid_ppl = valid_ppl + minibatch_valid_ppl * sum(lengths)

		valid_ppl = valid_ppl / sum(valid_lengths)

		print "\ntrain ppl: %f, valid ppl: %f" % (train_ppl, valid_ppl)

		if valid_ppl < min_valid_ppl:
			min_valid_ppl = valid_ppl
			model.save(args.save_net, vocab)
			if args.train_method != 'LSTM' :
				model.save_turing(args.save_net)
			stop_count = 0
			change_count = 0
			print "save best model"
			continue

		if args.early_stop:
			if (valid_ppl - min_valid_ppl) / min_valid_ppl > args.improvement_rate:
				if stop_count > 2 or lr < 1e-6:
					print 'stop training'
					break
				stop_count = stop_count + 1

		elif (valid_ppl - min_valid_ppl) / min_valid_ppl > args.improvement_rate * 0.5:
#			if change_count > 2:
			print 'change learning rate from %f to %f' % (lr, lr/2)
			model.lstm_lr = model.lstm_lr / 2.
			model.turing_lr = model.turing_lr / 2.
			model.all_lr = model.all_lr / 2.

			if args.train_method == 'LSTM' :
				lr         = model.lstm_lr
			elif args.train_method == 'TURING' :
				lr         = model.turing_lr
			else :
				lr         = model.all_lr
#			change_count = change_count + 1


def testing(args, test_data, test_lengths):

	print 'start loading'
	model_load = Model(
		input_size=1,
		hidden_size=1,
		vocab_size=1,
		stack_size=1, # make this bigger, but makes compilation slow
		celltype=RNN # use RNN or LSTM
	)

	model_load.stop_on(vocab.word2index["."])

	if args.train_method != 'LSTM' :
		if not os.path.isfile(args.load_net + '.turing') :
			print "there is no trained turing file so we can't test by turing model!!"
			sys.exit()

	model_load.load(args.load_net, 0)

	

	# test ppl
	#select correct train and prediction method according to train_method(LSTM/TURING/ALL)
	if args.train_method == 'LSTM' :
		ppl_fun    = model_load.lstm_ppl_fun
	else :
		ppl_fun    = model_load.final_ppl_fun

	minibatch_size = min(20, len(test_lengths))
	test_ppl = 0
	n_test_batches = len(test_lengths)/minibatch_size
	for minibatch_idx in range(n_test_batches):
		minibatch_test_data, lengths = get_minibatch(test_data, test_lengths, minibatch_size, minibatch_idx)
		minibatch_test_ppl = ppl_fun(minibatch_test_data, list(lengths))
		test_ppl = test_ppl + minibatch_test_ppl * sum(lengths)
		
	# last minibatch
	if (minibatch_idx + 1) * minibatch_size != len(test_lengths):
		minibatch_idx = minibatch_idx + 1
		n_rest_example = len(test_lengths) - minibatch_size * minibatch_idx
		minibatch_test_data, lengths = get_minibatch(test_data, test_lengths, n_rest_example, minibatch_idx)

		minibatch_test_ppl = ppl_fun(minibatch_test_data, list(lengths))
		test_ppl = test_ppl + minibatch_test_ppl * sum(lengths)
		
	test_ppl = test_ppl / sum(test_lengths)

	print "test ppl: %f" %test_ppl


if __name__ == "__main__":

	
	parser = argparse.ArgumentParser(description=DESCRIPTION)
	args = parse_args(parser)

	# if no args are passed
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()

	if args.train:
		vocab = build_vocab(args.train[0])
		train_data, train_lengths = load_data(args.train[0], vocab, 'train')
		valid_data, valid_lengths = load_data(args.valid[0], vocab, 'valid')
		training(args, vocab, train_data, train_lengths, valid_data, valid_lengths)

	elif args.test:
		vocab = pickle.load(open(args.load_net+'.vocab', "rb"))
		test_data, test_lengths = load_data(args.test[0], vocab, 'test')
		testing(args, test_data, test_lengths)

