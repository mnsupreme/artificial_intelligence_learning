#THIS IS NOT ORIGINAL CODE!!! THIS CODE WAS WRITTEN BY SIRAJ RAVAL, KRISTIAN WICHMANN and RANDALL BOHN. IT ONLY HAS MY EXTRA COMMENTARY!!

#
# ORIGINAL AUTHORS
#
# check them out! :)

# Siraj Raval github username: ||Source|| link to profile: https://github.com/llSourcell
#
# Kristian Wichmann github username: kwichmann link to profile: https://github.com/kwichmann
#
# Randall Bohn github username: rsbohn link to profile:https://github.com/rsbohn

import numpy as np

#sigmoid function to normalize data. turns any number to a number between 1 and  0
def nonlin(x,deriv=False):

    #this returns the derivative of sigmoid if "true" is passed in
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
x = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
expected = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
weights0 = 2*np.random.random((3,4)) - 1
weights1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    inputs = X
    # hidden layer 1 values is the dot product of the weights0 and input
    # turned into a sigmoid
    hidden_layer_values = nonlin(np.dot(inputs,weights0))
    # output layer 1 values is the dot product of the weights1 and input
    # turned into a sigmoid
    output = nonlin(np.dot(hidden_layer_values,weights1))

    # how much did we miss the target value?
    output_error = expected - output

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(output_error)))
        
    # assumed cost function is 0.5(output_error^2)
    # the function below is the derivative of the cost function in relation
    # to hidden_layer_1_values
    # the reason why we plug in output into the sigmoid function instead of
    # the dot product of weights0 and hidden_layer_1_values is because the 
    # derivative of a sigmoid y= 1/(1+(e^x)) is y(1-y) NOT x(1-x)
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
   change_in_output_values_error = output_error*nonlin(output,deriv=True)

    # how much did each hidden_layer value contribute to the output error (according to the weights)?
    # derivative of the cost function in respect to the hidden layer is "weights1 * (derivative of cost function in respect to the layer after it. In this case the outout layer) * sum(hidden_layer_values(1 - hidden_layer_values))"
    # below we have just taken out the "weigths1 * (derivative of cost function in respect to the last layer. In this case the output layer)" and set it to be our error
    # ".T" means transpose. it is transposing the matrix to be multiplied properly
    hidden_layer_values_error = change_in_output_values_error.dot(weights1.T)
    
    # in what direction is the target hidden_layer_values?
    # were we really sure? if so, don't change too much.
    # as you can see, we plugged in the hidden_layers_value_error to get the same equation mentioned earlier.
    # Here it is again: "weights1 * (derivative of cost function in respect to the layer after it. In this case the outout layer) * sum(hidden_layer_values(1 - hidden_layer_values))"
    change_in_hidden_layer_values_error = hidden_layer_values_error * nonlin(hidden_layer_values,deriv=True)

    # update the weights
    # ".T" means transpose it is tranposing the matrix to be multiplied properly
    weights1 += output.T.dot(change_in_output_values_error)
    weights0 += hidden_layer_values.T.dot(change_in_hidden_layer_values_error)

