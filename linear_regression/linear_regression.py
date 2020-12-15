#Roshan Shrestha

# Run
# python linear_regression.py training.txt degree lambda test.txt

import numpy as np 
import sys
#Read file and extract training samples and class labels
training = np.genfromtxt(str(sys.argv[1]))
m,n = np.shape(training)
columns = n-1
x = training[:,:n-1]
labels = training[:,n-1]
#Use command line arguments
degree = int(sys.argv[2])
lambda1 = int(sys.argv[3])
#Calculate phi(x)
phiofx = np.ones((m,1+columns*degree))
rowcount = 0
for row in x:
	phiindex = 1
	for i in range(columns):
		for j in range(degree):
			phiofx[rowcount,phiindex] = row[i]**(j+1)
			phiindex += 1
	rowcount += 1
#Calculate weights using least squares
I = np.identity(1+columns*degree) #Identity matrix for regularized least squares
weights1 = np.linalg.pinv(lambda1*I + np.dot(np.transpose(phiofx),phiofx)) #use np.linalg.pinv()
weights2 = np.dot(np.transpose(phiofx),labels)
weights = np.dot(weights1,weights2)
#Training phase output
for i in range(len(weights)):
	print("w%d=%.4f" % (i,weights[i]))
#Read test file
test = np.genfromtxt(str(sys.argv[4]))
m1,n1 = np.shape(test)
x = test[:,:n1-1]
labels = test[:,n1-1]
#Find target labels -> output
phiofx = np.ones((m1,1+columns*degree))
rowcount = 0
for row in x:
	phiindex = 1
	for i in range(columns):
		for j in range(degree):
			phiofx[rowcount,phiindex] = row[i]**(j+1)
			phiindex += 1
	rowcount += 1
#Test phase output
output = np.dot(phiofx,weights)
for i in range(m1):
	sumsquarederror = (labels[i] - output[i])**2
	print("ID=%5d, output=%14.4f, target value=%10.4f, squared error=%.4f" % (i+1,output[i],labels[i],sumsquarederror))
#Done