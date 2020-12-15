# Roshan Shrestha


import numpy as np
import sys
import math
from collections import Counter

def STANDARD_DEVIATION(examples, mean):
	N = len(examples)
	square_sum = 0
	for x in examples:
		square_sum += (x-mean)**2
	square_sum /= N
	return math.sqrt(square_sum)

def NORMALIZE(training, test):
	for j in range(len(training[0])-1):
		mean = sum(training[:,j])/len(training[:,j])
		std = STANDARD_DEVIATION(training[:,j], mean)
		for i in range(len(training)):
			training[i][j] -= mean
			training[i][j] /= std
		for i in range(len(test)):
			test[i][j] -= mean
			test[i][j] /= std

def EUCLIDEAN_DISTANCE(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return math.sqrt(distance)

def GET_NEIGHBORS(training, test_row, num_neighbors):
	distances = list()
	# distances is a list of tuples
	for training_row in training:
		distance = EUCLIDEAN_DISTANCE(test_row, training_row)
		distances.append((training_row, distance))
	distances.sort(key = lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def PREDICT_CLASSIFICATION(traininig, test_row, num_neighbors):
	neighbors = GET_NEIGHBORS(training, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key = output_values.count)
	return prediction

# ----------------------------------------------------------------------------

try:
	training_file = str(sys.argv[1])
	test_file = str(sys.argv[2])
	# k->num_neighbors
	k = int(sys.argv[3])
	if k < 1:
		sys.exit(1)
	if len(sys.argv) < 4:
		sys.exit(2)
except:
	print('python knn_classify.py <training_file> <test_file> <k>')

training = np.genfromtxt(training_file)
m1 = np.shape(training)[0]
test = np.genfromtxt(test_file)
m2 = np.shape(test)[0]

# check if there are no missing class labels like in satellite dataset
# there is no class label 6 in satellite dataset 
# but there is a class 7
# creates an error in this algorithm
# so make amends
# edit the freaking dataset itself
# editing training labels
for i in range(len(Counter(training[:,-1]))):
	if i in training[:,-1]:
		continue
	else:
		for j in range(m1):
			if training[j][-1] > i:
				training[j][-1] -= 1
# editing test labels
for i in range(len(Counter(test[:,-1]))):
	if i in test[:,-1]:
		continue
	else:
		for j in range(m2):
			if test[j][-1] > i:
				test[j][-1] -= 1

NORMALIZE(training, test)
correct = 0
ID = 0
for row in test:
	accuracy = 0
	predicted = PREDICT_CLASSIFICATION(training, row, k)
	if row[-1] == predicted:
		accuracy = 1
		correct += 1
	print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2lf\n'\
		%(ID, predicted, row[-1], accuracy))
	ID += 1
print('classification accuracy=%6.4f\n' % (correct*100.0/ID))

"""
References:
Machine Learning Mastery - Jason Brownlee
"""