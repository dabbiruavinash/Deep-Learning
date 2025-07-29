Create a Convolutional Netural Network and train it on mnist handwritten digits datasets. Using the model, find out the digit written by hand in a given image.

# hand written digit image recognition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np

# load mnist data for digit recognition from keras dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data

# display the shape of data
print('Shape of training data' , x_train.shape)
print('Shape of testing data = ', x_test.shape)
print('No. of Training samples = ', x_train.shape[0])
print('No. of Testing samples = ', x_test.shape[0])

# reshape the data into 4 dimensions as this is needed  by CNN
x_train = x_train.reshape(60000, 28, 28, 1) # 1 - no. of channels
x_test = x_test.reshape(10000, 28, 28, 1)

# convert targets (class vectors) to binary class metrices
from tensorflow.keras import utils
x_train = utils.to_categorical(y_train, 10) # there are 10 classes
y_test = utils.to_categorical(y_train, 10)

# display a sample train image array and label
arr = y_train[100] # take 100th image
print(arr) # [0. 0. 0. 0.0. 1. 0. 0. 0. 0.]

label = np.argmax(arr) # which digit this image shows?
print(label) # 5

# display the image
import matplotlib.pyplot as plt
plt.imshow(x_train[100], cmap = 'gray') # cmap for gray image

# build our CNN model
model = Sequential()
input_layer = Dense(32, input_shape = (28, 28, 1)
model.add(input_layer)

conv_layer1 = Conv2D(32, kernel_size= (3,3), activation = 'relu') # 3x3 cinvolution grid
model.add(conv_layer2)

conv_layer2 = Conv2D(64, (3,3), activation = 'relu')
model.add(conv_layer2)

pool_layer = MaxPooling2D(pool_size=(2,2))
model.add(pool_layer)

drop_layer = Dropout(0.5) # to avoid overfitting
model.add(flat_layer)

output_layer = Dense(10, activation = 'softmax') # final output 10 classes
mode.add(output_layer)

# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrices = ['accuracy'])

# train the model - this took 22 minutes on my laptop with 8 GB i7
history = model.fit(x_train, y_train, batch_size = 32, epochs = 10, verbose = 1, validation_data = (x_test, y_test__

# evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc) # 0.045, 0.987

# make prediction with 100th test data
arr = model.predict(([x_train[100].reshape(1,28,28,1)]]
print(arr)
label = np.argmax(arr)
print(label) # 5
