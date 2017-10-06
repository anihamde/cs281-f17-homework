import utils

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import optim
import time
import math

timebefore = time.clock()

BS = 1000
eta = 0.05

train_iter, val_iter, test_iter, text_field = utils.load_imdb(batch_size=BS)

def logistic_reg_torch(nfeats,nclasses):
	model = torch.nn.Sequential()
	model.add_module('logistic',torch.nn.Linear(nfeats, nclasses, bias=True))
	return model

def train(model,loss,optimizer,x,y):
	x = Variable(x)
	y = Variable(y)

	optimizer.zero_grad()

	forward_x = model.forward(x)
	obj = loss.forward(forward_x,y)

	obj.backward()

	optimizer.step()

	return obj.data[0]

def predict(model, x):
	# x = Variable(x)
	forward_x = model.forward(x)
	return forward_x.data.numpy().argmax(axis=1)


model = logistic_reg_torch(245703,2)
optimizer = optim.SGD(model.parameters(),lr=eta)
loss = torch.nn.CrossEntropyLoss()

print "TRAINING"

for k in range(0,100):
	timeb = time.clock()
	cost = 0.
	for batch in train_iter:
		xval = utils.bag_of_words(batch,text_field)
		yval = batch.label-1
		cost += train(model,loss,optimizer,xval.data,yval.data)
	predtrain = predict(model,xval)
	print time.clock() - timeb
	print "Epoch %d, cost = %f, acc = %.2f%%"%(k + 1, cost/(k+1), 100. * np.mean(predtrain == yval.data.numpy()))

print "VALIDATION"

for batch in val_iter:
	xval = utils.bag_of_words(batch,text_field)
	yval = batch.label-1
predval = predict(model,xval)
print "acc = %.2f%%"%(100. * np.mean(predval == yval.data.numpy()))

print "TEST"

for batch in test_iter:
	xval = utils.bag_of_words(batch,text_field)
	yval = batch.label-1
predtest = predict(model,xval)
print "acc = %.2f%%"%(100. * np.mean(predtest == yval.data.numpy()))

print time.clock()-timebefore
