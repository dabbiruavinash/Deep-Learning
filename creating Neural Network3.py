Using Python code, create a two layered Neural Network with relu() and sigmoid() activation functions

# create ReLU function gives x if x >= 0 else gives 0
def relu(x):
       return (max(0,x))

# create sigmoid function gives 1 if x > 0 else gives 0
def sigmoid(x):
       if x <= 0:
             return 0
       elseL
             return 1

# import numpy for creating arrays
import numpy as np
# Define weights for two Hidden Layers and one output layer.
weights = {'x2' : np.array([1,1]),
                  'x3' : np.array([-1,1]),
                  'x4' : np.array([-2,1]),
                  'x5' : np.array([1,0]),
                  'x6' : np.array([1.-1]) }

# assume the input data is an array of 2 elements
data = np.array([2,3])

# the input data should be given to nodes in the input layer
x0 = data[0]
x1 = data[1]

# create the input layer with the nodes x0 and x1
input_layer = np.array([x0,x1])
print("Input Layer = ", input_layer)

# The nodes in the Hidden layer1 receive:
# sum of (input from input layer * weights of Hidden layer1)
x2 = (input_layer * weights['x2']).sum()
x3 = (input_layer * weights['x3']).sum()

# pass the above values to ReLU function
x2 = relu(x2)
x3 = relu(x3)

# create the Hidden layer1 with nodes x2 and x3
hidden_layer1 = np.array([x2,x3])
print("Hidden Layer1 = ", hidden_layer1)

# The nodes in the Hidden layer2 receive:
# sum of (inputs from Hidden layer1 * weights of Hidden layer2)
x4 = (hidden_layer1 * weights['x4']).sum()
x5 = (hidden_layer1 * weights['x5']).sum()

# pass the above values to ReLU functions
x4 = relu(x4)
x5 = relu(x5)

# create the Hidden layer2 with the nodes x4 and x5
hidden_layer2 = np.array([x4,x5])
print("Hidden Layer2= ", hidden_layer2)

# The nodes in the output layer receive:
# sum of (input from Hidden layer2 * weights of output layer)
x6 = (hidden_layer2 * weight['x6']).sum()

# pass the above value to sigmoid function
x6 = sigmoid(x6)

# create output layer and display it
output_layer = np.array([x6])
print("output Layer = ", output_layer)