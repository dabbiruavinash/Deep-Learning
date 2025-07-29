Create ANN with 2 layers that accepts two input values and gives a single result.

# import numpy for creating arrays
import numpy as np

' ' ' create ANN with 4 layers.
Input layer - -> contains 2 nodes x0,x1
Hidden layer - -> contains 2 nodes x2,x3
Hidden layer - -> contains 2 nodes x4,x5
Output layer - -> contains 1 nodes x6
there will not be weights associated with Input layer.
so, let us define weights for hidden and output layers. ' ' '

weights = {'x2' : np.array([1,1]),
                  'x3' : np.array([-1,1]),
                  'x4' : np.array([-2,1]),
                  'x5' : np.array([1,0]),
                  'x6' : np.array([1.-1]) }

# assume the input data is an array of 2 elements
data = np.array([2,3])

# the input data should be given to nodes in the Input layer
x0 = data[0]
x1 = data[1]

# create the input layer with the nodes x0 and x1
input_layer = np.array([x0,x1])
print("Input Layer =", input_layer)

# the nodes in the Hidden layer1 receive:
# sum of (inputs from input layer * weights of Hidden layer1)
x2 = (input_layer * weights['x2']).sum()
x3 = (input_layer * weights['x3']).sum()

# create the Hidden layer1 with the nodes x2 and x3
hidden_layer1 = np.array([x2,x3])
print("Hidden Layer1 = ", hidden_layer1)

# The nodes in the Hidden layer2 receive:
# sum of (inputs from Hidden layer1 * weights of Hidden layer2)
x4 = (hidden_layer1 * weights['x4']).sum()
x5 = (hidden_layer1 * weights['x5']).sum()

# create the Hidden layer2 with the nodes x4 and x5
hidden_layer2 = np.array([x4,x5])
print("Hidden Layer2 = ", hidden_layer2)

# The nodes in the Output layer receive:
# sum of (input from Hidden layer2 * weights of output layer)
x6 = (hidden_layer2 * weight['x6']).sum()

# create output layer and display it
output_layer = np.array([x6])
print("Output Layer", output_layer)

