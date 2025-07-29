create a ANN and train it on house price dataset. Classify the house price is above average or below average.

# houseprice classification using ANN
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# import the dataset as a dataframe
df = pd.read_csv("E:/test/housepricedata.csv")
df.head()

# take independent features as x and target as y
x = df.iloc[:, 0:10]
y = df.iloc[:, 10]

# scale the data from 0 to 1
from sklearn.preprocessing import MinMaxSalar
x_scale = scaler.fit_transform(x)

# let us split the dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.3, random_state=1)

# creating ANN model
# create Sequential model so that the layers can be added one by one 
model = Sequential()

# create Input Lyer using predefined Dense Layer
input_layer = Dense(20, input_shape = (10,))

# add the Input Layer to the model
model.add(input_layer)

# create Hidden Layer using predefined Dense Layer
hidden_layer1 = Dense(20, activation= 'relu')

# add hidden layer to the model
model.add(hidden_layer1)

# create sedond hidden layer using predefined Dense Layer
hidden_layer2 = Dense(20, activation='relu')

# add Hidden Layer to the model
model.add(hidden_layer2)

# create output layer as a Dense Layer
# only one output that may be 0 or 1
output_layer = Dense(1, activation='sigmoid')

# add the output layer to the model
model.add(output_layer)

# compiling model will make the model ready to run in memory
model.compile(loss='binary=crossentropy', optimizer = 'sgd', metircs = ['accuracy'])

# train the model
model.fit(x_train, y_train, epochs = 100, batch_size=10)

# evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# make prediction the 10 values related to a house are given below
lst = [[14260, 8,5,1145,2,1,4,9,1,836]]

# scale the data
scaled_data = scaler.transform(lst)

result = model.predict(scaled_data)
if result >= 0.5:
   print('1')
else:
   print('0') 