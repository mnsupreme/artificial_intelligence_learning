#THIS IS NOT ORIGINAL CODE!!! THIS CODE WAS WRITTEN BY SIRAJ RAVAL AND WILSON MAR. IT ONLY HAS MY EXTRA COMMENTARY AND A FEW EXTRA FEATURES!!

#
# ORIGINAL AUTHORS
#
# check them out! :)

# Siraj Raval github username: ||Source|| link to profile: https://github.com/llSourcell
#
# Wilson Mar github username: wilsonmar link to profile: https://github.com/wilsonmar
 
import numpy as np
from numpy import *

def run():
	points = genfromtxt("data.csv", delimiter=",")
	learning_rate = .000165251 #adjust this to reduce error
	initial_m = 0
	initial_b = 0
	overfit_penalizer = 0.0000001 #This is also known as your regularization lambda
	num_iterations = 1000
	[final_m, final_b] = gradient_run(points,initial_m, initial_b, num_iterations, learning_rate, overfit_penalizer)
	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, get_error(initial_m, initial_b, points))
	print "Running..."
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, final_b, final_m, get_error(final_m, final_b, points))


def gradient_run(points, guess_m, guess_b, num_iterations, learning_rate, overfit_penalizer):
	current_m =  guess_m
	current_b = guess_b
	for i in range(num_iterations):
		current_m, current_b = gradient_step(current_m, current_b, learning_rate, overfit_penalizer, array(points))
	return [current_m, current_b]

def gradient_step(m_current,b_current,learning_rate,overfit_penalizer, points):
	m_gradient = 0
	b_gradient = 0
	N = float(len(points))
	x_average = np.average(points[:,0])
	x_range= np.ptp(points[:,0])
	for i in range (0, len(points)):
		x = (points[i, 0]-x_range)/x_average #feature scale (normalize) your inputs
		y = points[i, 1]

		#assume cost function is 1/(2N)*((y - ((m_current * x) + b_current))^2)
		b_gradient += -(1/N) * (y - ((m_current * x) + b_current)) # no need for overfit penalizer because b is y-intercept in y=mx+b
		m_gradient += -(1/N) * x * (y - ((m_current * x) + b_current)) + ((overfit_penalizer/N) * m_current)
		# add overfit_penalizer to prevent overfitting data
	new_m = m_current - (learning_rate * m_gradient)
	new_b = b_current - (learning_rate * b_gradient)
	#learning rate helps to speed up ir slowdown gradient adjustment. If set to high, it will overshoot optimum y and b values
	return [new_m, new_b]

def get_error(m,b,points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) ** 2
	return totalError / float(len(points))

if __name__ == '__main__':
    run()