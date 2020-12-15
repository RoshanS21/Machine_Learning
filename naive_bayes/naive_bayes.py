# Roshan Shrestha

# Run
# python .\naive_bayes.py .\training.txt .\_test.txt

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4)
import math
import sys

#Training
training = np.genfromtxt(str(sys.argv[1]))
m,n = np.shape(training)

#calculating no. of dimensions
dimension = n-1
no_of_class = 0
#calculating no. of classes
for i in range(m):
	if training[i][n-1] > no_of_class:
		no_of_class = int(training[i][n-1])

	
mean = np.empty([no_of_class,n-1], dtype = float)
#calculate mean
for i in range(no_of_class):
	for j in range(dimension):
		s = 0
		count = 0
		for k in range(m):
			if i+1 == training[k][n-1]:
				s += training[k][j]
				count += 1
		mean[i][j] = s/count
#std -> standard deviation
std = np.empty([no_of_class,n-1], dtype = float)
#calculate standard deviation
for i in range(no_of_class):
	for j in range(dimension):
		s = 0
		count = -1
		for k in range(m):
			if i+1 == training[k][n-1]:
				s += (training[k][j] - mean[i][j])**2
				count += 1
		#Any time the value for the standard deviation is computed to be smaller than 0.01,
		#your code should replace that value with 0.01.
		std[i][j] = (s/count)**(1/2) if ((s/count)**(1/2)) > 0.01 else 0.01
#Printing out mean and std for each class and dimension
for i in range(no_of_class):
		for j in range(dimension):
			print("Class %d, attribute %d, mean = %.2f, std = %.2f" %(i+1,j+1,mean[i][j],std[i][j]))

test = np.genfromtxt(str(sys.argv[2]))

pC = np.empty([no_of_class,1], dtype = float)
for i in range(no_of_class):
	count = 0
	for j in range(m):
		if i+1 == training[j][n-1]:
			count += 1
	pC[i] = count/m

def compute_Gaussian(no_of_class,dimension,x,std,mean):
	tempG = np.empty([no_of_class,dimension], dtype = float)
	for j in range(no_of_class):
		for k in range(dimension):
			tempG[j][k] = (1/(std[j][k]*math.sqrt(2*np.pi)))*math.exp(-1*(math.pow(x[k]-mean[j][k],2))/(2*math.pow(std[j][k],2)))
			#tempG[j][k] = (1/(std[j][k]*((2*math.pi)**(1/2))))*math.exp(-1*((x-mean[j][k])**2)/(2*(std[j][k])**2))
	return tempG

#Classification
input_number = 0
total_accuracy = 0
for input in test:
	Gaussian = compute_Gaussian(no_of_class,dimension,input,std,mean)
	#P(X|C)
	PXgivenC = np.empty([no_of_class,1], dtype = float)
	for i in range(no_of_class):
		temp = 1
		for j in range(dimension):
			temp *= Gaussian[i][j]
		PXgivenC[i][0] = temp
	#P(X)
	PX = 0
	for i in range(no_of_class):
		PX += PXgivenC[i][0] * pC[i]
	#P(C|X) = P(X|C)*p(C)/P(X)
	PCgivenX = np.empty([no_of_class,1], dtype = float)
	for i in range(no_of_class):
		PCgivenX[i][0] = PXgivenC[i][0] * pC[i] / PX
	#find Largest probability class in P(C|X)
	class_index = 1
	tie = 1
	greatest = PCgivenX[0][0]
	tie_class = []
	for i in range(no_of_class):
		if PCgivenX[i][0] > greatest:
			greatest = PCgivenX[i][0]
			class_index = i+1
		if PCgivenX[i][0] == greatest:
			tie += 1
			tie_class.append(i+1)
	#Calculating accuracy
	accuracy = 0
	if tie == 1 and class_index == int(input[dimension]):
		accuracy = 1
	elif tie == 1 and class_index != int(input[dimension]):
		accuracy = 0
	elif tie > 1 and int(input[dimension]) in tie_class:
		accuracy = 1 / len(tie_class)
	else:
		accuracy = 0
	total_accuracy += accuracy
	input_number = input_number + 1
	print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (input_number,class_index,greatest,int(input[dimension]),accuracy))
print("classification accuracy=%6.4f" %(total_accuracy/input_number*100))