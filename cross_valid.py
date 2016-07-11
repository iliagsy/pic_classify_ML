# coding:utf-8
from preprocessing import *
import numpy as np
import theano
import theano.tensor as T

# training_set, test_set format: tuple(input, target)
# input is a numpy.ndarray of 2 dimensions (a matrix)
# where each row corresponds to an example. target is a
# numpy.ndarray of 1 dimension (vector) that has the same length as
# the number of rows in the input. It should give the target
# to the example with the same index in the input.
def shared_dataset(data_xy, borrow=True):
	""" Function that loads the dataset into shared variables

	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""
	data_x, data_y = data_xy
	shared_x = theano.shared(np.asarray(data_x,
										   dtype=theano.config.floatX),
							 borrow=borrow)
	shared_y = theano.shared(np.asarray(data_y,
										   dtype=theano.config.floatX),
							 borrow=borrow)
	# When storing data on the GPU it has to be stored as floats
	# therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations
	# we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning
	# ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
	return shared_x, T.cast(shared_y, 'int32')

sub_sets = []

for i in range(10):
	sub_set = sum( [ sub_data_set[ len(sub_data_set)*i/10 : len(sub_data_set)*(i+1)/10 ] 
		for sub_data_set in data_set ], [] )
	sub_sets.append( sub_set )

test_set_No = 0

def get_data_sets( test_set_No ):

	training_set = sum( [sub_sets[i] for i in range(10) if i != test_set_No ], [] )
	training_set = [(img.flatten(),label)  for img,label in training_set]
	training_set_x = np.array( [x  for x,y in training_set] )
	training_set_y = np.array( [y  for x,y in training_set] )
	training_set = (training_set_x, training_set_y)

	test_set = sub_sets[ test_set_No ]
	test_set = [(img.flatten(),label)  for img,label in test_set]
	test_set_x = np.array( [x  for x,y in test_set] )
	test_set_y = np.array( [y  for x,y in test_set] )
	test_set = (test_set_x, test_set_y)

	test_set_x, test_set_y = shared_dataset( test_set )
	training_set_x, training_set_y = shared_dataset( training_set )

	datasets = [(training_set_x, training_set_y), (test_set_x, test_set_y)]
	
	return datasets
	
datasets = get_data_sets( test_set_No )
	

	