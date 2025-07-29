create an ANN using TensorFlow and Keras libraries and use it for classification of Iris flowers using the dataset available in scikit-learn.org site

# Iris flowers classification using ANN
import tensorflow as tf
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense

# import iris flowers dataset from scikit-learn.org
from sklearn.datasets import load_iris
iris = load_iris()

# display the column names of the dataset
dir(iris)

# display the feature names
iris.feature_names

# display the input data
iris.data

# display the target column names
iris.target_names

# display the data in the target column
iris.target

# let us split the dataset into training and testing data 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state=1)

# creating ANN model
# create sequential model so that the layers can be added one by one type the following 7 lines in one cell
model = Sequential()

# create input layer using predefined Dense Layer
input_layer = Dense(4, input_shape = (4,))

# add the Input Layer to the model
mode.add(input_layer)

# create Hidden layer using predefined Dense Layer
hidden_layer = Dense(4, activation='relu')

# add Hidden Layer to the model
model.add(hidden_layer)

# create output layer as a Dense Layer
# the output belongs to 3 classes (0,1 or 2) and hence 3 nodes
output_layer = Dense(3, activation='softmax')

# add the output layer to the model
model.add(output_layer)

# compiling model will make the model ready to run in memory
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adm', metrics = ['accuracy'])

# train the model -this will take some time
model.fit(x_train, y_train, epochs=300, batch_size=4)

# evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# make prediction
import numpy as np
arr = np.array([[5.1, 3.5, 1.4, 0.2]]) # find out which flower it is
result = model.predict(arr)
print(np.argmax(result)) # 0 -> it is setosa