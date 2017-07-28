#This code solves for the optimum values of the parameters analytically using calculus. 
# It is an alternative way to optimizing iterarively using gradient descent. It is usually faster
# but is much more computationally expensive. It is good if you have 1000 or less parameters to solve for.
# complexity is O(n^3)

# This examply assumes the equation y = param_0(y-intercept) + param_1 * input_1 + param_2 * input_2....
# If you have an equation in a different form such as y = param_0(y-intercept) + param_1 * input_1 * input_2 + param_2 * (input_2)^2
# then you must make adjustments to the design matrix accordignly. (Multiply the inputs acordingly and use those results in the design matrix)

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm 
import numpy as np
from numpy.linalg import inv

data = np.loadtxt('gradient_descent/ex3Data/ex3x.dat',dtype=float)

#this function helps put the inputs in vector form as the original data already came as a design matrix. 
#This way the data is more realistic. Technically, this function is unecessary
def vectorize_data():
	#regular python list
	vectorized_data = []
	for row in data:
		# inserts a 1 in the beginning every row of the matrix to represent x_0 input
		row = np.insert(row,0,1)
		print "row {0}, \n row_transpose {1}".format(row, row.reshape((-1,1)))
		#converts [element1, element 2] to [        form.
		#								element1,
		#								element2,
		#									]
		vectorized_data.append(row.reshape((-1,1)))
	#you need to convert design matrix into a numpy array because it starts out as a regular python list.
	vectorized_data = np.array(vectorized_data)
	print vectorized_data
	return vectorized_data

x = vectorize_data()

#converts vectorized data into a design matrix. 
# The design matrix looks like this: design_matrix = [transpose_of_each_input_vector or transpose_of_each_row_in_vectorized_data]
def assemble_design_matrix():
	#regulare python list
	design_matrix = []
	for row in x:
		design_matrix.append(row.transpose())
	#you need to convert design matrix into a numpy array before you convert it to a matrix because it starts out asa regular python list.
	design_matrix = np.matrix(np.array(design_matrix))
	print "design_matrix {0} \n design_matrix_transpose {1}".format(design_matrix, design_matrix.transpose())
	return design_matrix

def normal():
	design_matrix = assemble_design_matrix()
	y = np.loadtxt('gradient_descent/ex3Data/ex3y.dat', dtype=float)
	y = y.reshape(47,1)
	# THIS IS THE NORMAL EQUATION FORMULA
	# the function is inverse(design_matrix_transpose * design_matrix) * (design_matrix_transpose * y_vector)
	# this will yield a matrix full of your optimized theta parameter values
	result = inv((design_matrix.transpose() * design_matrix)) * (design_matrix.transpose() * y)
	print result
	return result


if __name__ == '__main__':
    normal()