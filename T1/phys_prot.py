from __future__ import division
import numpy as np
import math
import torch
from torch import Tensor
from torch.autograd import Variable
import time

def normalize(X):
	params = []
	ncols = len(X[0])

	for i in range(ncols):
		feat = X[:,i]
		mu = np.mean(feat)
		sig = np.std(feat)
		params.append([mu,sig])

		feat = (feat - mu)/sig

		X[:,i] = feat

	return X, params

def normalize_test(X,params):
	ncols = len(X[0])

	for i in range(ncols):
		feat = X[:,i]
		mu = params[i][0]
		sig = params[i][1]

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
trainX,trainparams = normalize(trainX)
testX = normalize_test(testX,trainparams)

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

print (w_ridg)

predic_test = np.matmul(w_ridg,np.transpose(testX))
preRMSE = testy - predic_test

print(((1/len(preRMSE))*sum(np.multiply(preRMSE,preRMSE)))**(0.5))

#################
# Begin Problem 5
print("#################################")

#w = np.zeros(10)
#weights = Variable(Tensor(w), requires_grad=True)
weights = Variable(torch.randn(10,1),requires_grad=True)

optimizer = torch.optim.LBFGS([weights])

def black_box():
	# optimizer.zero_grad()
	weights_data = weights.data.numpy()
	
	# print(   ((((np.reshape(trainy,(trainy.shape[0],1)) - np.dot(trainX,weights_data)))**2)).shape )
	# print((((np.reshape(trainy,(trainy.shape[0],1)) - np.dot(trainX,weights_data)))**2).sum())
	print("trainy",trainy)
	print("trainX",trainX)
	print("weights",weights_data)

	objective = 0.5*((((np.reshape(trainy,(trainy.shape[0],1)) - np.dot(trainX,weights_data)))**2).sum() + 10.0*(weights_data**2).sum())
	deriv = np.dot(-1*trainX.T, (np.reshape(trainy,(trainy.shape[0],1)) - np.dot(trainX,weights_data))) + 10.0*weights_data

	# objective = sum((trainy - np.matmul(weights_data,np.transpose(trainX)))**2)/2 + (10)*(1/2)*np.matmul(weights_data,np.transpose(weights_data))
	# deriv = -np.matmul(trainX.T,(trainy - np.dot(trainX,weights_data))) + 10*weights_data
	# -np.matmul(np.transpose(trainX),trainy - np.matmul(weights_data,np.transpose(trainX))) + 10*weights_data
	weights.grad = Variable(Tensor(deriv))

	return Variable(Tensor([objective]))
	# return objective

for i in range(100):
	ws = optimizer.step(black_box)
	print(ws)

print(ws)

print("RMSE:",((1/len(preRMSE))*((np.dot(testX,weights.data.numpy())-testy)**2).sum())**(1/2))

#################
# Begin Problem 6

print("###############")

def nonlin(d):
	A = np.random.normal(size=(d,10))
	b = np.random.uniform(high=2*np.pi,size=(d,1))

	trainXnon=np.cos(np.matmul(A,trainX.T)+b)
	testXnon=np.cos(np.matmul(A,testX.T)+b)

	return trainXnon.T,testXnon.T



# 4, 5 applied to 6

# 100
time_start = time.time()
trainXnon, testXnon = nonlin(100)
Q,R = np.linalg.qr(trainXnon)
Rinv = np.linalg.inv(R)
w_ridg = np.matmul(np.matmul(Rinv,np.transpose(Q)),trainy)

predic_test = np.matmul(w_ridg,np.transpose(testXnon))
preRMSE = testy - predic_test
print(((1/len(preRMSE))*sum(np.multiply(preRMSE,preRMSE)))**(0.5))

# print(w_ridg)
# print(preRMSE)
running_time = time.time() - time_start
print("QR_100",running_time)




# 200
time_start = time.time()
trainXnon, testXnon = nonlin(200)
Q,R = np.linalg.qr(trainXnon)
Rinv = np.linalg.inv(R)
w_ridg = np.matmul(np.matmul(Rinv,np.transpose(Q)),trainy)

predic_test = np.matmul(w_ridg,np.transpose(testXnon))
preRMSE = testy - predic_test
print(((1/len(preRMSE))*sum(np.multiply(preRMSE,preRMSE)))**(0.5))

# print(w_ridg)
# print(preRMSE)
running_time = time.time() - time_start
print("QR_200",running_time)




# 400
time_start = time.time()
trainXnon, testXnon = nonlin(400)
Q,R = np.linalg.qr(trainXnon)
Rinv = np.linalg.inv(R)
w_ridg = np.matmul(np.matmul(Rinv,np.transpose(Q)),trainy)

predic_test = np.matmul(w_ridg,np.transpose(testXnon))
preRMSE = testy - predic_test
print(((1/len(preRMSE))*sum(np.multiply(preRMSE,preRMSE)))**(0.5))

# print(w_ridg)
# print(preRMSE)
running_time = time.time() - time_start
print("QR_400",running_time)




# 600
time_start = time.time()
trainXnon, testXnon = nonlin(600)
Q,R = np.linalg.qr(trainXnon)
Rinv = np.linalg.inv(R)
w_ridg = np.matmul(np.matmul(Rinv,np.transpose(Q)),trainy)

predic_test = np.matmul(w_ridg,np.transpose(testXnon))
preRMSE = testy - predic_test
print(((1/len(preRMSE))*sum(np.multiply(preRMSE,preRMSE)))**(0.5))

# print(w_ridg)
# print(preRMSE)
running_time = time.time() - time_start
print("QR_600",running_time)







