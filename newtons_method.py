import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm 
import numpy as np
from numpy.linalg import inv

fig = plt.figure(figsize=plt.figaspect(0.5))

newton_ax = fig.add_subplot(1,2,1, projection='3d')
answer_ax = fig.add_subplot(1,2,2, projection='3d')

inputs = np.insert(np.loadtxt('gradient_descent/ex4Data/ex4x.dat',dtype=float),0,1,axis=1)

print inputs
expected = np.loadtxt('gradient_descent/ex4Data/ex4y.dat', dtype=float)
expected = expected.reshape(-1,1)
print expected
num_iterations = 10
N = float(inputs.shape[0])

def sigmoid(x, params):
	return 1/(1+np.exp(x * -params))


def newton():
	params = np.matrix([[0],[0],[0]], dtype=float)
	for d in range(num_iterations):
		h_x = sigmoid(inputs, params)
		print h_x
		error = (h_x - expected)
		jacobian = error.T*inputs*1/N
		print "inputs squared {0}, h_x mult{1}, jacobian {2}, h_x {3}".format(np.dot(inputs.T,inputs).shape, np.diag(h_x.A1).shape, jacobian.shape, h_x.shape)
		hessian = 1/N * inputs.T * np.matrix(np.diag(h_x.A1),dtype=float) * np.matrix(np.diag((1-h_x).A1),dtype=float) * inputs
		print hessian
		params = params - (inv(hessian) * jacobian.T)
	print params
	return params

def test(params):
	test_1 = (20 - 37.85)/9.8610344285
	test_2 = (80-67.38125)/9.82161002267
	print "test_1 {0}, test_2 {1}".format(test_1,test_2)
	print np.matrix(([1,test_1,test_2]), dtype=float) * params 

def map(params):
	X_1 = inputs[:,1]
	X_2 = inputs[:,2]
	Z = np.copy(expected).T
	newton_ax.scatter(X_1,X_2,Z,s=10,c="red")
	answer_ax.scatter(X_1,X_2,Z,s=10,c="red")
	H_X_newton = sigmoid(inputs,params).A1
	H_X_answer = sigmoid(inputs,np.matrix([[-16.38],[0.1483],[0.1589]], dtype=float)).A1
	X_1,X_2 = np.meshgrid(X_1,X_2)
	newton_ax.plot_surface(X_1,X_2,H_X_newton,cmap=cm.coolwarm)
	answer_ax.plot_surface(X_1,X_2,H_X_answer,cmap=cm.coolwarm)
	plt.show()
	fig.tight_layout()


test(newton())
map(newton())
