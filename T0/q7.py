import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
	return math.cos(x) + x**2 + math.exp(x)

def grad_f(x):
	return -math.sin(x) + 2*x + math.exp(x)

def grad_check(x,epsilon):
	return ((f(x+epsilon)-f(x-epsilon))/(2*epsilon))

je0 = [2,1,0.5,0.1,0.01]

je = np.array([je0,je0,je0,je0,je0,je0,je0])

jout = np.zeros([7,5])

xvals = [-100,-10,-1,0,1,10,100]

for i in range(0,7):
	for k in range(0,5):
		jout[i][k] = grad_check(xvals[i],je[i][k])

for i in range(0,len(xvals)):
	plt.plot(je[i],jout[i]-grad_f(xvals[i]),'bo')
	plt.title("x = %s"%xvals[i])
	plt.ylabel('Difference between analytic and numeric')
	plt.xlabel('Epsilon')
	# plt.show()
	plt.savefig("x%s.png"%xvals[i],format="png")
	plt.clf()