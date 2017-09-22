from __future__ import division
import numpy as np
import math
import torch
from torch import Tensor
from torch.autograd import Variable

def normalize(X):
	ncols = len(X[0])

	for i in range(ncols):
		feat = X[:,i]
		mu = np.mean(feat)
		sig = np.std(feat)

		feat = (feat - mu)/sig

		X[:,i] = feat

	return X

data = np.loadtxt("CASP.csv", delimiter = ",", skiprows = 1)
y = data[:,0]
X = data[:,1:]

nrows = len(data)
nrows_train = int(math.ceil(0.9*nrows))

# Separate training data
trainX = X[:nrows_train,:]
trainy = y[:nrows_train]

# Separate test data
testX = X[nrows_train:,:]
testy = y[nrows_train:]

# Normalize
trainX = normalize(trainX)
testX = normalize(testX)

# trainY = (trainy-np.mean(trainy))/np.std(trainy)
# testY = (testy-np.mean(testy))/np.std(testy)

# Add bias feature
trainX = np.c_[trainX,np.ones(nrows_train)]
testX = np.c_[testX,np.ones(nrows - nrows_train)]

#################
# Begin Problem 4

Q,R = np.linalg.qr(trainX)
Rinv = np.linalg.inv(R)
w_ridg = np.matmul(np.matmul(Rinv,np.transpose(Q)),trainy)

predic_test = np.matmul(w_ridg,np.transpose(testX))
preRMSE = testy - predic_test

# print ((1/len(preRMSE))*sum(np.multiply(preRMSE,preRMSE)))**(0.5)

#################
# Begin Problem 5

weights = Variable(Tensor(10),requires_grad=True)

optimizer = torch.optim.LBFGS(weights)

def black_box():
	weights_data = weights.data.numpy()

	objective = sum(((testy - np.matmul(weights_data,np.transpose(testX)))**2)/2) + (10)*(1/2)*np.matmul(weights_data,np.transpose(weights_data))

	weights.grad = Tensor({numpy})

	return {objective}

for i in range(100):
	optimizer.step(black_box)


#################
# Begin Problem 6






