from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d, relu
from theano.gradient import grad

from cross_valid import datasets, get_data_sets
from preprocessing import dim_vals, dim_val

error_on_train = []
error_on_test = []

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
		
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        
        if y.dtype.startswith('int'):
            
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        
        self.input = input
        
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]

class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(4, 3)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        pooled_out = pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=False
        )

        self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input


def evaluate_lenet5(datasets_=datasets, learning_rate=[17./(3**i) for i in range(6)], n_epochs=42,
                    nkerns=[12, 12, 0, 0], batch_size=1,
                    patience=200000, filter_shape=[3, 3, 0],
                    poolsize=[2, 2, 0] ):
	
	rng = numpy.random.RandomState(23455)

	train_set_x, train_set_y = datasets_[0]
	test_set_x, test_set_y = datasets_[1]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches //= batch_size
	n_test_batches //= batch_size

	index = T.lscalar()  # index to a [mini]batch

	x = T.matrix('x')   # the data is presented as images
	y = T.ivector('y')  # the labels are presented as 1D vector of
						# [int] labels

	######################
	# BUILD ACTUAL MODEL #
	######################
	print('... building the model')

	image_shape = (batch_size, dim_vals[0], dim_vals[1], dim_vals[2])
	
	layer0_input = x.reshape( image_shape )
	
	# Construct the first convolutional pooling layer:
	# filtering reduces the image size to (264-5+1 , 264-5+1) = (260, 260)
	# maxpooling reduces this further to (260/2, 260/2) = (130, 130)
	# 4D output tensor is thus of shape (batch_size, nkerns[0], 130, 130)
	layer0 = LeNetConvPoolLayer(
		rng,
		input=layer0_input,
		image_shape=image_shape,
		filter_shape=(nkerns[0], dim_vals[0], filter_shape[0], filter_shape[0]),
		poolsize=(poolsize[0], poolsize[0])
	)
	
	# Construct the second convolutional pooling layer
	# filtering reduces the image size to (130-3+1, 130-3+1) = (128, 128)
	# maxpooling reduces this further to (128/2, 128/2) = (64, 64)
	# 4D output tensor is thus of shape (batch_size, nkerns[1], 64, 64)
	layer1_input_shape = (dim_val+1-filter_shape[0]) / poolsize[0]
	layer1 = LeNetConvPoolLayer(
		rng,
		input=layer0.output,
		image_shape=(batch_size, nkerns[0], layer1_input_shape, layer1_input_shape),
		filter_shape=(nkerns[1], nkerns[0], filter_shape[1], filter_shape[1]),
		poolsize=(poolsize[1], poolsize[1])
	)
	
	# the HiddenLayer being fully-connected, it operates on 2D matrices of
	# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
	# This will generate a matrix of shape (batch_size, nkerns[3] * 31 * 31)
	layer2_input = layer1.output.flatten(2)

	layer2_input_shape = (layer1_input_shape+1-filter_shape[1]) / poolsize[1]
	layer2 = HiddenLayer(
		rng,
		input=layer2_input,
		n_in=nkerns[1] * layer2_input_shape * layer2_input_shape,
		n_out=500,
		activation=T.tanh
	)
	
	# the HiddenLayer being fully-connected, it operates on 2D matrices of
	# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
	# This will generate a matrix of shape (batch_size, nkerns[3] * 31 * 31)
	# layer3_input = layer2.output
# 
# 	layer3 = HiddenLayer(
# 		rng,
# 		input=layer3_input,
# 		n_in=1000,
# 		n_out=500,
# 		activation=T.tanh
# 	)
	
	layer4 = LogisticRegression(input=layer2.output, n_in=500, n_out=13)

	cost = layer4.negative_log_likelihood(y)

	test_model = theano.function(
		[index],
		layer4.errors(y),
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	test_model_on_train = theano.function(
		[index],
		layer4.errors(y),
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	params = layer4.params + layer2.params + layer1.params + layer0.params

	grads = grad(cost, params)

	updates_0 = [
		(param_i, param_i - learning_rate[0] * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	updates_1 = [
		(param_i, param_i - learning_rate[1] * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	updates_2 = [
		(param_i, param_i - learning_rate[2] * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	updates_3 = [
		(param_i, param_i - learning_rate[3] * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	updates_4 = [
		(param_i, param_i - learning_rate[4] * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	updates_5 = [
		(param_i, param_i - learning_rate[5] * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	# updates_6 = [
# 		(param_i, param_i - learning_rate[6] * grad_i)
# 		for param_i, grad_i in zip(params, grads)
# 	]
	
	train_model_0 = theano.function(
		[index],
		cost,
		updates=updates_0,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	train_model_1 = theano.function(
		[index],
		cost,
		updates=updates_1,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	train_model_2 = theano.function(
		[index],
		cost,
		updates=updates_2,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	train_model_3 = theano.function(
		[index],
		cost,
		updates=updates_3,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	train_model_4 = theano.function(
		[index],
		cost,
		updates=updates_4,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	train_model_5 = theano.function(
		[index],
		cost,
		updates=updates_5,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	# train_model_6 = theano.function(
# 		[index],
# 		cost,
# 		updates=updates_6,
# 		givens={
# 			x: train_set_x[index * batch_size: (index + 1) * batch_size],
# 			y: train_set_y[index * batch_size: (index + 1) * batch_size]
# 		}
# 	)
	

	###############
	# TRAIN MODEL #
	###############
	print('... training')
	# early-stopping parameters
	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False
	cblr = [n_epochs*i/len(learning_rate) for i in range(len(learning_rate)+1)]
	for i in range( len(learning_rate) ):
		while (epoch in range(cblr[i],cblr[i+1])) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in range(n_train_batches):

				iter = (epoch - 1) * n_train_batches + minibatch_index

				if iter % 1000 == 0:
					print('training @ iter = ', iter)
				if i == 0:
					cost_ij = train_model_0( minibatch_index )
				elif i == 1:
					cost_ij = train_model_1( minibatch_index )
				elif i == 2:
					cost_ij = train_model_2( minibatch_index )
				elif i == 3:
					cost_ij = train_model_3( minibatch_index )
				elif i == 4:
					cost_ij = train_model_4( minibatch_index )
				elif i == 5:
					cost_ij = train_model_5( minibatch_index )
				# elif i == 6:
# 					cost_ij = train_model_6( minibatch_index )

				if patience <= iter:
					done_looping = True
					break

	end_time = timeit.default_timer()
	print('Optimization complete.')
	print(('The code for file ' +
		   os.path.split(__file__)[1] +
		   ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
	# test it on the training set
	test_losses = [
		test_model_on_train(i)
		for i in range(n_train_batches)
	]
	test_score = numpy.mean(test_losses)
	print(('     epoch %i, minibatch %i/%i, test error on training set '
		   '%f %%') %
		  (epoch, minibatch_index + 1, n_train_batches,
		   test_score * 100.))
	error_on_train.append( test_score * 100. )
		   
	# test it on the test set
	test_losses = [
		test_model(i)
		for i in range(n_test_batches)
	]
	test_score = numpy.mean(test_losses)
	print(('     epoch %i, minibatch %i/%i, test error '
		   '%f %%') %
		  (epoch, minibatch_index + 1, n_train_batches,
		   test_score * 100.))
	error_on_test.append( test_score * 100. )

# cross-validation
if __name__ == '__main__':
	for test_set_No in range(10):
		datasets = get_data_sets( test_set_No )
		evaluate_lenet5( datasets )	
	Mean_error_on_test = numpy.mean( error_on_test )
	Mean_error_on_train = numpy.mean( error_on_train )
	print('mean error rate on train =', Mean_error_on_train )
	print('mean error rate on test =', Mean_error_on_test )
	print('max on test =', numpy.max( error_on_test ) )