import numpy as np
import utils
import math
import itertools
import time

BS = 174000

train3,val3,test3 = utils.load_jester(load_text=False,batch_size=BS,subsample_rate=0.1)

timebefore = time.time()

num_epochs = 100
sigmasqeps = 1.0

trainlogliks = np.zeros((10,num_epochs))
testlogliks = np.zeros((10,num_epochs))

for K in range(1,2):
    U = np.random.normal(0,5,(70000,K))
    V = np.random.normal(0,5,(150,K))

    for batch in train3:
        for testbatch in test3:
            trainratings = batch.ratings-1
#             trainratings = trainratings.type(torch.FloatTensor)
            trainusers = batch.users-1 
            trainjokes = batch.jokes-1


            testratings = testbatch.ratings-1
#             testratings = testratings.type(torch.FloatTensor)
            testusers = testbatch.users-1 
            testjokes = testbatch.jokes-1
        
            trainratings = trainratings.data.numpy()
            testratings = testratings.data.numpy()
            trainusers = trainusers.data.numpy()
            testusers = testusers.data.numpy()
            trainjokes = trainjokes.data.numpy()
            testjokes = testjokes.data.numpy()

            print "batch loaded"
            
            lavec = np.logical_and(np.isin(testusers,trainusers),np.isin(testjokes,trainjokes))
            
            print "batch loaded 2"
            
            testratings = testratings[lavec]
            testusers = testusers[lavec]
            testjokes = testjokes[lavec]

            for epoch in range(0,num_epochs):
                for user in trainusers: # in range(0,len(U))
                    uservec = (trainusers == user)
                    Ri = trainratings[uservec]
                    Vi = V[trainjokes[uservec]]
                    sigmaprU = np.linalg.inv((0.2)*np.identity(K)+(1/sigmasqeps)*np.dot(Vi.transpose(),Vi))
                    muprU = np.dot(sigmaprU,(1/sigmasqeps)*np.dot(Vi.transpose(),Ri))

                    U[user] = np.random.multivariate_normal(muprU,sigmaprU)

                print "fin users"
                
                print time.time()-timebefore
                
                for joke in trainjokes: # in range(0,len(V))
                    jokesvec = (trainjokes == joke)

                    Rj = trainratings[jokesvec]
                    Uj = U[trainusers[jokesvec]]

                    sigmaprV = np.linalg.inv((0.2)*np.identity(K)+(1/sigmasqeps)*np.dot(Uj.transpose(),Uj))
                    muprV = np.dot(sigmaprV,(1/sigmasqeps)*np.dot(Uj.transpose(),Rj))

                    V[joke] = np.random.multivariate_normal(muprV,sigmaprV)

                print "fin jokes"
                
                print time.time()-timebefore
                    
                Utrain = U[trainusers]
                Utest = U[testusers]
                Vtrain = V[trainjokes]
                Vtest = V[testjokes]            

                trainlogliks[K-1][epoch] = -np.sum(np.power(trainratings-np.sum(Utrain*Vtrain,1),2))/(2*sigmasqeps)
                testlogliks[K-1][epoch] = -np.sum(np.power(testratings-np.sum(Utest*Vtest,1),2))/(2*sigmasqeps)
                
                print "done with epoch %s for K=%s"%(epoch+1,K)
                
    print time.time()-timebefore

    print trainlogliks
    print testlogliks