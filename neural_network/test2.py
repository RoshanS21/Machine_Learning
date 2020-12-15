# Roshan Shrestha
# 1001555263

import numpy as np
import sys
from math import exp
from collections import Counter
# seeding so that we can check the output
# if no seeding is done, we cannot compare the random values
#np.random.seed(1)

training_file = "pendigits_training.txt"
test_file = "pendigits_test.txt"
layers = 3
units_per_layer = 3
rounds = 10

# Read file and extract training samples and class labels
training = np.genfromtxt(training_file)
m1,n1 = np.shape(training)
training_x = training[:,:n1-1]
training_labels = training[:,n1-1]

# Read file and extract test samples and class labels
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
# max_value for normalizing
max_value = np.amax(training_x)
# Normalize training and test values
training_x /= max_value
test_x /= max_value

no_of_classes = len(Counter(training_labels))
# adding bias_input 1 to both training and test arrays
training_x = np.concatenate((np.ones((m1,1)), training_x), axis = 1)
test_x = np.concatenate((np.ones((m2,1)), test_x), axis = 1)

# Random weight between -0.05 and 0.05
def get_random():
	return np.random.uniform(low = -0.05, high = 0.05, size = None)

# Initialize network
# network is a list of list of dictionary
def initialize_network(n_inputs, n_layers, n_units, n_outputs):
	n_hidden = n_layers - 2
	network = list()
	# what if no hidden layers
	if n_hidden == 0:
		output_layer = [{'weights':[get_random() for j in range(n_inputs)]} for j in range(n_outputs)]
		network.append(output_layer)
		return network
	# if hidden layers
	for i in range(n_hidden):
		if i == 0:
			hidden_layer = [{'weights':[get_random() for j in range(n_inputs)]} for j in range(n_units)]
			network.append(hidden_layer)
		else:
			hidden_layer = [{'weights':[get_random() for j in range(n_units + 1)]} for j in range(n_units)]
			network.append(hidden_layer)
	output_layer = [{'weights':[get_random() for j in range(n_units + 1)]} for j in range(n_outputs)]
	network.append(output_layer)
	return network


# calculate activation, a = (w.T * x) for an input
def activate(weights, inputs):
	activation = 0.0
	for i in range(len(weights)):
		activation += weights[i] * inputs[i]
	return activation

# Activation function, sigmoid, z = h(a) = sigma(a)
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []	#store output of current layer
		for neuron in layer:
			# a = (w.T * x)
			activation = activate(neuron['weights'], inputs)
			# sigmoid
			# also save this output in the network for current neuron for future access
			neuron['output'] = transfer(activation)
			# store output of this layer for
			# input of next layer
			new_inputs.append(neuron['output'])
		# insert 1 at position 0 as bias input
		# insert 1 to match the dimension of w and x for w.T * x
		new_inputs.insert(0, 1)

		inputs = new_inputs
	return inputs[1:len(inputs)+1]	# output deosn't have a bias value

# Backpropagate error and store in neurons
# basically, I am calculating the gradient here
# Remember, output layer gradient is different from hidden layer gradient
def backward_propagate_error(network, expected):
	# reversed sends the loop in reverse order
	# by reverse order, i mean the value of i decreases from len(network)-1 to 0
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:	# hidden layer gradient
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:	# output layer gradient
			for j in range(len(layer)):
				neuron = layer[j]
				gradient = neuron['output'] - expected[j]
				errors.append(gradient)
		# appending gradients for weight update
		# weight updates will be performed by a separate method
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * ((neuron['output']) * (1.0 - (neuron['output'])))

# Update network weights with error i.e gradient and learning_rate
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row
		if i != 0:
			# inputs is updated for every layer that is not first hidden layer
			inputs = [neuron['output'] for neuron in network[i - 1]]
			inputs.insert(0,1)	# bias input 1
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]

# Train a network for a fixed number of epochs i.e. rounds
def train_network(network, train, training_labels, l_rate, n_epoch, n_outputs):
	m = np.shape(train)[0]
	for epoch in range(n_epoch):
		for index in range(m):
			row = train[index]
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[int(training_labels[index])] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		# l_rate starts with 1
		# for each round, multiply l_rate with 0.98
		l_rate *= 1


# Testing
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs

# Make a prediction with a network
def classification_accuracy(network, m2):
	total_accuracy = 0.0
	for i in range(m2):
		accuracy = 0.0
		outputs = predict(network, test_x[i])
		classes = Counter(outputs)
		predicted_class = outputs.index(max(outputs))
		true_class = int(test_labels[i])
		# check accuracy
		if predicted_class == true_class and classes[outputs[predicted_class]] == 1:
			accuracy = 1.0
			total_accuracy += accuracy
		elif predicted_class == true_class and classes[outputs[predicted_class]] > 1:
			accuracy = 1.0 / classes[outputs[predicted_class]]
			total_accuracy += accuracy
		else:
			accuracy = 0.0
	return total_accuracy / m2 * 100

def test(layers, units_per_layer, rounds):
	network = initialize_network(np.shape(training_x)[1], layers, units_per_layer, no_of_classes)
	# Training the network
	train_network(network, training_x, training_labels, 1, rounds, no_of_classes)
	acc = classification_accuracy(network,m2)
	print('layers=%4d, units_per_layer=%4d, rounds=%4d, classification accuracy=%6.4f\n' % (layers, units_per_layer, rounds, acc))


test(3,20,20)


"""
References
Machine Learning Mastery
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""