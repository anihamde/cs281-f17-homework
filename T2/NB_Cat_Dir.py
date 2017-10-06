import numpy as np
import utils
import pandas as pd
import time
import math

timebefore = time.clock()

def probclass(listv,powerlist):
	p_arr = np.array(map(math.log,listv))
	power_arr = np.array(power_arr)

	return np.dot(p_arr,power_arr)


	# multval = 1
	# for i in range(0,len(listv)):
	# 	multval = multval*(listv[i]**powerlist[i])

	# return multval

alpha = [1,0.8,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
phat_MLE = pd.DataFrame(np.zeros([2,245703]))
BS = 1000

train_iter, val_iter, test_iter, text_field = utils.load_imdb(batch_size=BS)

print "It took",time.clock()-timebefore,"seconds to load the data!"

N = 0
for batch in train_iter:
	x = utils.bag_of_words(batch,text_field)
	y = batch.label - 1
	y = y.data.numpy()
	N += np.sum(x.data.numpy())

	xdata = x.data.numpy()
	xdata = [xdata[y==k] for k in [0,1]]
	xdata[0] = np.sum(xdata[0],axis=0)
	xdata[1] = np.sum(xdata[1],axis=0)

	phat_MLE.iloc[0] = phat_MLE.iloc[0] + xdata[0]
	phat_MLE.iloc[1] = phat_MLE.iloc[1] + xdata[1]

print "It took",time.clock()-timebefore,"seconds to train Naive Bayes!"

# Validation

correct = [0.0]*len(alpha)
incorrect = [0.0]*len(alpha)

phat_MLE_fin = [0]*len(alpha)

for j in range(0,len(alpha)):
	phat_MLE_fin[j] = (phat_MLE+alpha[j])/(N+alpha[j]+0.0)
	phat_MLE_fin[j].to_csv('NB_Beta_Bern_%s.csv'%alpha[j])

for batch in val_iter:
	x = utils.bag_of_words(batch,text_field)
	y = batch.label-1
	y = y.data.numpy()

	for j in range(0,len(alpha)):
		for i in range(0,len(x.data.numpy())):
			pclass0 = probclass(phat_MLE_fin[j].iloc[0],x.data.numpy()[i])
			pclass1 = probclass(phat_MLE_fin[j].iloc[1],x.data.numpy()[i])

			if np.argmax(np.array([pclass0,pclass1])) == int(y[i]):
				correct[j] += 1
			else:
				incorrect[j] += 1

print "VALIDATION RESULTS*********************************"
for j in range(0,len(alpha)):
	print "For alpha = %s, there were %s correct predictions and %s incorrect predictions, meaning %s classification accuracy."%(alpha[j],correct[j],incorrect[j],correct[j]/incorrect[j])

print "It took",time.clock()-timebefore,"seconds to validate Naive Bayes!"

# Test

correct = [0.0]*len(alpha)
incorrect = [0.0]*len(alpha)

for batch in test_iter:
	x = utils.bag_of_words(batch,text_field)
	y = batch.label-1
	y = y.data.numpy()

	for j in range(0,len(alpha)):
		for i in range(0,len(x.data.numpy())):
			pclass0 = probclass(phat_MLE_fin[j].iloc[0],x.data.numpy()[i])
			pclass1 = probclass(phat_MLE_fin[j].iloc[1],x.data.numpy()[i])

			if np.argmax(np.array([pclass0,pclass1])) == int(y[i]):
				correct[j] += 1
			else:
				incorrect[j] += 1

print "TEST RESULTS*********************************"
for j in range(0,len(alpha)):
	print "For alpha = %s, there were %s correct predictions and %s incorrect predictions, meaning %s classification accuracy."%(alpha[j],correct[j],incorrect[j],correct[j]/incorrect[j])

print "It took",time.clock()-timebefore,"seconds to test Naive Bayes!"

print "It took",time.clock()-timebefore,"seconds to implement Naive Bayes!"
