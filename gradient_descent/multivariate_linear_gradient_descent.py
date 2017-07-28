import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm 
import numpy as np

fig = plt.figure(figsize=plt.figaspect(0.5))

batch_ax = fig.add_subplot(2,2,1, projection='3d')
classic_stochastic_ax = fig.add_subplot(2,2,2, projection='3d')
mini_batch_ax = fig.add_subplot(2,2,3, projection='3d')
answer_ax = fig.add_subplot(2,2,4, projection='3d')
def feature_scale(column):
	print "column {0}".format(column.reshape(-1,1))
	feature_average = np.mean(column)
	feature_range = np.std(column)
	print "feature_average {0} feature_range {1}".format(feature_average,feature_range)
	scaled_features = np.matrix(np.copy(column)).reshape(-1,1)
	print "unscaled features {0}".format(scaled_features)
	scaled_features = (scaled_features - feature_average)/feature_range
	print scaled_features
	return scaled_features.reshape(1,-1)

inputs = np.insert(np.loadtxt('ex3Data/ex3x.dat',dtype=float),0,1,axis=1)
print "inputs {0}".format(inputs)
print "inputs shape {0}".format(inputs.shape)
expected = np.loadtxt('ex3Data/ex3y.dat', dtype=float)
expected = expected.reshape(47,1)
print "expected {0}".format(expected)
print "expected.shape {0}".format(expected.shape)
num_iterations = 100
N = float(inputs.shape[0])
scaled_inputs = np.copy(inputs)
# normalize all inputs except for x_0 because x_0 is always 1
scaled_inputs[:,1] = feature_scale(inputs[:,1]) 
scaled_inputs[:,2] = feature_scale(inputs[:,2])
print "scaled features {0} \n scaled features shape {1}".format(scaled_inputs, scaled_inputs.shape)
print "scaled input row {0}".format(scaled_inputs[0])

def batch():
	current_parameters = np.matrix([[0],[0],[0]], dtype=float) 
	learning_rate = 1
	#overfit_penalizer = 0.00000001
	for i in range(num_iterations):
		#alternate form:
		#current_parameters = current_parameters - (learning_rate/N) * (scaled_inputs.T * ((scaled_inputs * current_parameters) - expected))
		current_parameters = current_parameters - ((learning_rate/N) * (((scaled_inputs * current_parameters) - expected).T * scaled_inputs).T)
		print "error {0}".format(learning_rate/N * (((scaled_inputs * current_parameters) - expected).T * scaled_inputs).T)
	print current_parameters
	return current_parameters

def classic_stochastic():
	current_parameters = np.matrix([[0],[0],[0]], dtype=float) 
	learning_rate = 1
	#overfit_penalizer = 0.00000001
	for j in range(num_iterations):
		for i in range(scaled_inputs.shape[0]):
			print "scaled inputs {0}".format(scaled_inputs[i].reshape(3,1))
			current_parameters = current_parameters - learning_rate/N * (((scaled_inputs[i] * current_parameters) - expected[i]).item() * scaled_inputs[i].reshape(3,1))
	print current_parameters
	return current_parameters

def mini_batch():
	current_parameters = np.matrix([[0],[0],[0]], dtype=float) 
	learning_rate = 1
	gradient = np.matrix([[0],[0],[0]], dtype=float)
	#overfit_penalizer = 0.00000001
	for j in range(num_iterations):
		for i in range(scaled_inputs.shape[0]):
			if i == 47 or i % 20 == 0:
				current_parameters = current_parameters - gradient
				gradient = np.matrix([[0],[0],[0]], dtype=float)
				print "current parameters {0}".format(current_parameters)
			gradient += learning_rate/N * (((scaled_inputs[i] * current_parameters) - expected[i]).T * scaled_inputs[i]).T
			print "gradient {0}".format(gradient)
	print current_parameters
	return current_parameters

def test(params):
	#square_ft = 1650
	#rooms = 3
	square_ft = (1650 - 2000.68085106)/786.202618743 #scaling input
	rooms = (3 - 3.17021276596)/0.752842809062 #scaling input
	print "square_ft {0} \n rooms {1}".format(square_ft,rooms)
	calc_cost(params)
	print np.matrix(([1,square_ft,rooms]), dtype=float) * params

def calc_cost(params):
	error = ((scaled_inputs * params) - expected)
	cost = 1/(2*N) * (error.T * error)
	print "cost {0}".format(cost)

def map(params_batch,params_classic_stochastic,params_mini_batch):
	X_1 = scaled_inputs[:,1]
	X_2 = scaled_inputs[:,2]
	Z = np.copy(expected).T
	batch_ax.scatter(X_1,X_2,Z,s=10,c="red")
	classic_stochastic_ax.scatter(X_1,X_2,Z,s=10,c="red")
	mini_batch_ax.scatter(X_1,X_2,Z,s=10,c="red")
	answer_ax.scatter(X_1,X_2,Z,s=10,c="red")
	H_X_batch = (scaled_inputs * params_batch).A1
	H_X_classic_stochastic =  (scaled_inputs * params_classic_stochastic).A1
	H_X_mini_batch = (scaled_inputs * params_mini_batch).A1
	H_X_answers = (scaled_inputs * np.matrix([[340413],[110631],[-6649]],dtype=float)).A1
	X_1,X_2 = np.meshgrid(X_1,X_2)
	print "H_X {0}".format(H_X_batch)
	batch_ax.plot_surface(X_1,X_2,H_X_batch,cmap=cm.coolwarm)
	classic_stochastic_ax.plot_surface(X_1,X_2,H_X_classic_stochastic,cmap=cm.coolwarm)
	mini_batch_ax.plot_surface(X_1,X_2,H_X_mini_batch,cmap=cm.coolwarm)
	answer_ax.plot_surface(X_1,X_2,H_X_answers,cmap=cm.coolwarm)
	plt.show()
	fig.tight_layout()

test(batch())
map(batch(),classic_stochastic(),mini_batch())
