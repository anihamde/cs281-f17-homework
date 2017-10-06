import numpy as np
import utils
import pandas as pd
import time

timebefore = time.clock()

indic = lambda x: 1 if x > 0 else 0

def probclass(listv,powerlist):
	p_arr = np.array(map(math.log,listv))
	power_arr = np.array(power_arr)

	return np.dot(p_arr,power_arr)

alpha = [1,0.8,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
beta = [1,0.8,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
phat_MLE = pd.DataFrame(np.zeros([2,245703]))
BS = 1000

train_iter, val_iter, test_iter, text_field = utils.load_imdb(batch_size=BS)

print "It took",time.clock()-timebefore,"seconds to load the data!"

# Training

N = 0
phat_MLE_fin = phat_MLE
for batch in train_iter:
	x = utils.bag_of_words(batch,text_field)
	y = batch.label - 1
	y = y.data.numpy()
	
	xindic = np.sign(x.data.numpy()[i])# map(indic,x.data.numpy()[i])
	N += np.sum(xindic)
	ind = int(y)
	for rew in range(0,len(ind)):
		indval = ind[rew]
		phat_MLE.iloc[indval] = phat_MLE.iloc[indval] + xindic[rew]
	print "hi"

print "It took",time.clock()-timebefore,"seconds to train Naive Bayes!"

# Validation

correct = [0.0]*len(alpha)
incorrect = [0.0]*len(alpha)

phat_MLE_fin = [0]*len(alpha)

for j in range(0,len(alpha)):
	phat_MLE_fin[j] = (phat_MLE+alpha[j])/(N+alpha[j]+beta[j]+0.0)
	phat_MLE_fin[j].to_csv('NB_Beta_Bern_%s.csv'%alpha[j])

for batch in val_iter:
	x = utils.bag_of_words(batch,text_field)
	y = batch.label-1
	y = y.data.numpy()

	for j in range(0,len(alpha)):
		xindc = np.sign(x.data.numpy())
		pclass0 = map(np.dot,xindic,np.log(phat_MLE_fin[j][0]))
		pclass1 = map(np.dot,xindic,np.log(phat_MLE_fin[j][1]))

		probs = pclass1 - pclass0
		compars = map(indic,probs)

		for k in range(0,len(compars)):
			if y[k] == compars[k]:
				correct[j] += 1
			else:
				incorrect[j] += 1
			# xindic = map(indic,x.data.numpy()[i])
			# pclass0 = probclass(phat_MLE_fin[j].iloc[0],xindic)
			# pclass1 = probclass(phat_MLE_fin[j].iloc[1],xindic)

			# if np.argmax(np.array([pclass0,pclass1])) == int(y[i]):
			# 	correct[j] += 1
			# else:
			# 	incorrect[j] += 1

print "VALIDATION RESULTS*********************************"
for j in range(0,len(alpha)):
	print "For alpha = beta = %s, there were %s correct predictions and %s incorrect predictions, meaning %s classification accuracy."%(alpha[j],correct[j],incorrect[j],correct[j]/incorrect[j])

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
			xindic = map(indic,x.data.numpy()[i])
			pclass0 = probclass(phat_MLE_fin[j].iloc[0],xindic)
			pclass1 = probclass(phat_MLE_fin[j].iloc[1],xindic)

			if np.argmax(np.array([pclass0,pclass1])) == int(y[i]):
				correct[j] += 1
			else:
				incorrect[j] += 1

print "TEST RESULTS*********************************"
for j in range(0,len(alpha)):
	print "For alpha = beta = %s, there were %s correct predictions and %s incorrect predictions, meaning %s classification accuracy."%(alpha[j],correct[j],incorrect[j],correct[j]/incorrect[j])

print "It took",time.clock()-timebefore,"seconds to test Naive Bayes!"

print "It took",time.clock()-timebefore,"seconds to implement Naive Bayes!"
