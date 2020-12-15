# Roshan Shrestha


import numpy as np
import sys
import math
from collections import Counter

#np.random.seed(1)

options = ['optimized', 'randomized', 'forest3', 'forest15']

try:
	# import files
	training_file = str(sys.argv[1])
	test_file = str(sys.argv[2])
	option = str(sys.argv[3])
	if option not in options:
		print('Option can be:')
		print(options)
		sys.exit('Try one of these options')
	pruning_thr = int(sys.argv[4])
	if pruning_thr < 0:
		sys.exit('pruning_thr cannot be less than 0')
except:
	sys.exit('python decision_tree.py <training_file> <test_file> <option> <pruning_thr>')

# Read file and extract training samples and class labels
training = np.genfromtxt(training_file)
m1,n1 = np.shape(training)

# Read file and extract test samples and class labels
test = np.genfromtxt(test_file)
m2,n2 = np.shape(test)

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

"""
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
"""

no_of_classes = len(Counter(training[:,-1]))


#---------------------------------------------------------------------------

def ENTROPY(Ki, K):
	# math formulae
	return -(Ki*1.0/K)*math.log(Ki*1.0/K, 2)

def NODE_ENTROPY(N, K):
	entropy = 0
	for i in N.keys():
		entropy += ENTROPY(N[i], K)
	return entropy


def SELECT_COLUMN(examples, attribute_column):
	return list(row[attribute_column] for row in examples)

def TEST_SPLIT(attribute_column, threshold, examples):
	left, right = list(), list()
	for row in examples:
		if row[attribute_column] < threshold:
			left.append(row)
		else:
			right.append(row)
	return left, right

def INFORMATION_GAIN(examples, attribute_column, threshold):
	left, right = TEST_SPLIT(attribute_column, threshold, examples)
	N1 = Counter(list(row[-1] for row in examples))
	K = np.shape(examples)[0]
	left_CL, right_CL = list(), list()
	for row in left:
		left_CL.append(row[-1])
	for row in right:
		right_CL.append(row[-1])
	N2 = Counter(left_CL)
	N3 = Counter(right_CL)
	info_gain = NODE_ENTROPY(N1,K) - (len(left_CL)/K) * NODE_ENTROPY(N2,len(left_CL)) - (len(right_CL)/K) * NODE_ENTROPY(N3,len(right_CL))
	return info_gain

# Optimize CHOOSE-ATTRIBUTE
# L -> smallest value of attribute A
# M -> largest value of attribute A
def OPTIMIZED_CHOOSE_ATTRIBUTE(examples, attributes):
	max_gain = best_threshold = -1
	index = -1
	for attribute_column in range(len(examples[0])-1):
		# attribute_values is the array containing the values of all examples
		# for attribute A
		attribute_values = SELECT_COLUMN(examples, attribute_column)
		L = min(attribute_values)
		M = max(attribute_values)
		# we are trying 50 threshold values between the min and max
		# 50 is just what we chose to go with
		for K in range(1,51):
			threshold = L+K*(M-L)/51
			gain = INFORMATION_GAIN(examples, attribute_column, threshold)
			if gain > max_gain:
				max_gain = gain
				best_threshold = threshold
				index = attribute_column
	groups = TEST_SPLIT(index, best_threshold, examples)
	return {'feature_ID':index, 'threshold': best_threshold, 'groups': groups, 'gain': max_gain}

def RANDOM_ELEMENT(attributes):
	random_element = np.random.randint(0,len(attributes)-1)
	return random_element

#Randomized CHOOSE-ATTRIBUTE for Decision Forests
def RANDOMIZED_CHOOSE_ATTRIBUTE(examples, attributes):
	max_gain = best_threshold = -1
	attribute_column = RANDOM_ELEMENT(attributes)
	attribute_values = SELECT_COLUMN(examples, attribute_column)
	L = min(attribute_values)
	M = max(attribute_values)
	for K in range(1,51):
		threshold = L+K*(M-L)/51
		gain = INFORMATION_GAIN(examples, attribute_column, threshold)
		if gain > max_gain:
			max_gain = gain
			best_threshold = threshold
	groups = TEST_SPLIT(attribute_column, best_threshold, examples)
	return {'feature_ID':attribute_column, 'threshold': best_threshold, 'groups': groups, 'gain': max_gain}

def SIZE(examples):
	return len(examples)

# DISTRIBUTION returns an array whose i-th position is the 
# probability of the i-th class
def DISTRIBUTION(examples):
	m,n = np.shape(examples)
	pC = []
	for i in range(no_of_classes):
		count = 0
		for j in range(m):
			if i == training[j][n-1]:
				count += 1
		pC.append(count*1.0/m)
	return pC

def CHECK_FOR_SAME_CLASS(examples):
	class_labels = list(row[-1] for row in examples)
	c = Counter(class_labels)
	if len(c) == 1:
		return 1
	else:
		return 0


def DTL(examples, attributes, default, pruning_thr, ID, tree_ID):
	if SIZE(examples) < pruning_thr:
		temp =  max(default)
		result = default.index(temp)
		return {'tree_ID': tree_ID, 'node_ID': ID, 'feature_ID': -1,\
			'threshold': -1, 'gain': 0, 'class': result, 'default': default}
	elif CHECK_FOR_SAME_CLASS(examples):
		return {'tree_ID': tree_ID, 'node_ID': ID, 'feature_ID': -1,\
			'threshold': -1, 'gain': 0, 'class': examples[0][-1], 'default': default}
	else:
		if option == 'optimized':
			tree = OPTIMIZED_CHOOSE_ATTRIBUTE(examples, attributes)
		else:
			tree = RANDOMIZED_CHOOSE_ATTRIBUTE(examples, attributes)
		examples_left = tree['groups'][0]
		examples_right = tree['groups'][1]
		del(tree['groups'])
		dist = DISTRIBUTION(examples)
		tree['tree_ID'] = tree_ID
		tree['node_ID'] = ID
		tree['left_child'] = DTL(examples_left, attributes, dist, pruning_thr, 2*ID, tree_ID)
		tree['right_child'] = DTL(examples_right, attributes, dist, pruning_thr, 2*ID+1, tree_ID)
		return tree


def DTL_TopLevel(examples, pruning_thr, tree_ID):
	# examples is the training data. It is a matrix,
	# where each row is a training object,
	# each column is an attribute,
	# the last column contains class labels.

	# attributes = list of all attributes in the examples
	attributes = examples[0]
	default = DISTRIBUTION(examples)
	return DTL(examples, attributes, default, pruning_thr, 1, tree_ID)

def PRINT_TREE(tree):
	if 'left_child' in tree:
		print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n'\
			%(tree['tree_ID'],tree['node_ID'],tree['feature_ID'] + 1,tree['threshold'],tree['gain']))
		PRINT_TREE(tree['left_child'])
		PRINT_TREE(tree['right_child'])
	else:
		print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n'\
			%(tree['tree_ID'],tree['node_ID'],tree['feature_ID'] + 1,tree['threshold'],tree['gain']))

# ----------------------------------------------------------------------------
# Classification

def PREDICT(tree, row):
	if row[tree['feature_ID']] < tree['threshold']:
		if 'left_child' in tree:
			return PREDICT(tree['left_child'], row)
		else:
			return tree
	else:
		if 'right_child' in tree:
			return PREDICT(tree['right_child'], row)
		else:
			return tree

def PREDICTIONS(tree, test):
	predictions = list()
	for row in test:
		predict_tree = PREDICT(tree, row)
		prediction = predict_tree['class']
		predictions.append(prediction)
	return predictions

def PRINT_CLASSIFICATION(test, predictions):
	total_accuracy = 0
	object_id = 1
	for row in test:
		accuracy = 0
		if row[-1] == predictions[object_id-1]:
			accuracy = 1
			total_accuracy += accuracy
		print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n'\
			%(object_id, predictions[object_id-1], row[-1], accuracy))
		object_id += 1
	total_accuracy = total_accuracy * 100.0 / m2 
	print('classification accuracy=%6.4f\n' % (total_accuracy))

if option == 'optimized':
	tree = DTL_TopLevel(training, pruning_thr, 1)
	PRINT_TREE(tree)
	predictions = PREDICTIONS(tree, test)
	PRINT_CLASSIFICATION(test, predictions)
elif option == 'randomized':
	tree = DTL_TopLevel(training, pruning_thr, 1)
	PRINT_TREE(tree)
	predictions = PREDICTIONS(tree, test)
	PRINT_CLASSIFICATION(test, predictions)
elif option == 'forest3':
	trees = list()
	three_predictions = list()
	for i in range(1,3+1):
		tree = DTL_TopLevel(training, pruning_thr, i)
		PRINT_TREE(tree)
		trees.append(tree)
		predictions = list()
		for row in test:
			predict_tree = PREDICT(tree, row)
			prediction = predict_tree['default']
			predictions.append(prediction)
		three_predictions.append(predictions)
	average_predictions = list()
	for j in range(len(three_predictions[0])):
		default1 = three_predictions[0][j]
		default2 = three_predictions[1][j]
		default3 = three_predictions[2][j]
		avg_default = list()
		total = 0
		for i in range(len(default1)):
			avg_default.append((default1[i]+default2[i]+default3[i])/3.0)
		highest = max(avg_default)
		average_predictions.append(avg_default.index(highest))
	PRINT_CLASSIFICATION(test, average_predictions)
elif option == 'forest15':
	fifteen_predictions = list()
	for i in range(1,15+1):
		tree = DTL_TopLevel(training, pruning_thr, i)
		PRINT_TREE(tree)
		predictions = list()
		for row in test:
			predict_tree = PREDICT(tree, row)
			prediction = predict_tree['default']
			predictions.append(prediction)
		fifteen_predictions.append(predictions)
	average_predictions = list()
	for j in range(len(fifteen_predictions[0])):
		default1 = fifteen_predictions[0][j]
		default2 = fifteen_predictions[1][j]
		default3 = fifteen_predictions[2][j]
		default4 = fifteen_predictions[3][j]
		default5 = fifteen_predictions[4][j]
		default6 = fifteen_predictions[5][j]
		default7 = fifteen_predictions[6][j]
		default8 = fifteen_predictions[7][j]
		default9 = fifteen_predictions[8][j]
		default10 = fifteen_predictions[9][j]
		default11 = fifteen_predictions[10][j]
		default12 = fifteen_predictions[11][j]
		default13 = fifteen_predictions[12][j]
		default14 = fifteen_predictions[13][j]
		default15 = fifteen_predictions[14][j]
		
		avg_default = list()
		total = 0
		for i in range(len(default1)):
			avg_default.append((default1[i]+default2[i]+default3[i]+\
				default4[i]+default5[i]+default6[i]+default7[i]+default8[i]+\
				default9[i]+default10[i]+default11[i]+default12[i]+default13[i]+\
				default14[i]+default15[i])/15.0)
		highest = max(avg_default)
		average_predictions.append(avg_default.index(highest))
	PRINT_CLASSIFICATION(test, average_predictions)