# Roshan Shrestha
# 1001555263

import numpy as np
import sys
from math import exp
from collections import Counter

#Random weight between -0.05 and 0.05
def get_random():
	return np.random.uniform(low = -0.05, high = 0.05, size = None)

# Initialize network
# network is a list of list of dictionary
def initialize_network(n_inputs, n_hidden, n_units, n_outputs):
	network = list()
	for i in range(n_hidden):
		hidden_layer = [{'weights':[get_random() for j in range(n_inputs + 1)]} for j in range(n_units)]
		network.append(hidden_layer)
	output_layer = [{'weights':[get_random() for j in range(n_units + 1)]} for j in range(n_outputs)]
	network.append(output_layer)
	return network

# calculate activation for an input
def activate(weights, inputs):
	activation = weights[0]	# because 1 is the bias input value of position 0 of input
	for i in range(1, len(weights)):
		activation += weights[i] * inputs[i]
	return activation

# Activation function
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		new_inputs.insert(0, 1)	# insert 1 at position 0 as bias input
		inputs = new_inputs
	return inputs[1:len(inputs)+1]	# output deosn't have a bias value

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
# basically, I am calculating the gradient here
# Remember, output layer gradient is different from hidden layer gradient
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:	# hidden layer gradient
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				neuron = layer[j]
				error *= neuron['output']
				errors.append(error)
		else:	# output layer gradient
			for j in range(len(layer)):
				neuron = layer[j]
				gradient = neuron['output'] - expected[j]
				gradient *= neuron['output']
				errors.append(gradient)
		# appending gradients for weight update
		# weight updates will be performed by a separate method
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error i.e gradient
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
			inputs.insert(0,1)	# bias input 1
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]

def insert_bias_input(row):
	np.insert(row,0,1)
	return row

# Train a network for a fixed number of epochs i.e. rounds
def train_network(network, train, training_labels, l_rate, n_epoch, n_outputs):
	m = np.shape(train)[0]
	for epoch in range(n_epoch):
		for index in range(m):
			row = train[index]
			row = insert_bias_input(row)	# bias input 1
			print(row)
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[training_labels[index]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		# l_rate starts with 1
		# for each round, multiply l_rate with 0.98
		l_rate *= 0.98


#-----------------------main-----------------------------------

training_file = "pendigits_training.txt"
test_file = "pendigits_test.txt"
layers = 3	# no. of layers. cannot be less than 2
if layers < 2:
	print("Number of layers >= 2")
	exit(-1)
units_per_layer = 2	# no. of perceprtrons at each hidden layer
rounds = 10	# training rounds, number of times to update weights
error_threshold = -1	# error is not the stopping criterion when it is -1
# if error is to be the stopping criterion, use 0.00001

#Read file and extract training samples and class labels
training = np.genfromtxt(training_file)
m1,n1 = np.shape(training)
training_x = training[:,:n1-1]
training_labels = training[:,n1-1]

#Read file and extract test samples and class labels
test = np.genfromtxt(test_file)
m2,n2 = np.shape(test)
test_x = test[:,:n2-1]
test_labels = test[:,n2-1]

# check if all labels start with 0
# if not, make them start with 0
zero_flag = 0
for i in range(m1):
	if int(training_labels[i]) == 0:
		zero_flag = 1
		break
if zero_flag == 0:
	for i in range(m1):
		training_labels[i] -= 1
	for i in range(m2):
		test_labels[i] -= 1

max_value = np.amax(training_x)
no_of_classes = len(Counter(training_labels))

# Normalize training and test values
training_x /= max_value
test_x /= max_value

network = initialize_network(n1-1, layers - 2, units_per_layer, no_of_classes)
# learning rate starts with 1
train_network(network, training_x, training_labels, 1, rounds, no_of_classes)

for layer in network:
	print(layer)
	print()




"""
References
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""