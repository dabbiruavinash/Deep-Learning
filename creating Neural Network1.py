write a python program to create a single layered neural network with 2 input values.

# for creating array of np
import numpy as np

' ' ' create ANN with 3 layers.
input layer --> contains 2 nodes x0,x1
hidden layer --> contains 2 nodes x2,x3
output layer --> contains 1 node x4
There will not be weights associated with the actual input data. so, define weights for hidden and output layers. ' ' '

weights = {'x2' : np.array([1,1]),
                  'x3' : np.array([-1,1]),
                  'x4' : np.array([2,-1]) }

# assume the input data is an array of 2 elements
data = np.array([2,3])

# the input data should be given to nodes in the input layer
x0 = data[0]
x1 = data[1]

# create the input layer with the nodes x0 and x1
input_layer = np.array([x0,x1])
print("Input Layer = ", input_layer)

# the nodes in the hidden layer receive
# sum of (inputs * weight of hidden layer)
x2 = (input_layer * weights['x2']).sum()
x3 = (input_layer * weights['x3']).sum()

# create the hidden with nodes x2 and x3 
hidden_layer = np.array([x2,x3])
print("Hidden Layer = ", hidden_layer)
# the nodes in the Output layer receive
x4 = (hidden_layer * weights['x4']).sum()

# create output layer and display iit
output_layer = np.array([x4])
print("Output Layer = ", output_layer)
