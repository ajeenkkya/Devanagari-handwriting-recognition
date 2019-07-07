#importing essential modules from library
import numpy as np
import pandas as pd
from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K

#fetching data using pandas
data = pd.read_csv("data.csv")
#converting it to numpy array to make the matrix operations easier
dataset = np.array(data)
#shuffling the dataset
np.random.shuffle(dataset)
X = dataset
Y = dataset
#sclicing only the first 1023 columns so that we would get our features
x = X[:, 0:1024]
#sclicing the 1024th column which has the label for every record
y = Y[:, 1024]
#as the dataset is shuffled so selecting first 70K tuples won't harm
x_train = x[0:70000, :]
#diving by 255 makes it easier for learning data as now the value is in between 0 and 1(basically normalization)
x_train = x_train/255

x_test = x[70000:72001, :]
#adjusting the shape of the matrices
y = y.reshape(y.shape[0], 1)
y_train = y[0:70000, :]
y_train = y_train.T
y_test = y[70000:72001, :]
y_test = y_test.T

#we are taking an image of 32x32 for training purpose
image_x = 32
image_y = 32
#to_categorical is an function which converts a matrix to a vector; in our case it basically gets the label which is an integer
#and changes it to form of one_hot_encoding.
#Check this link for more details->https://keras.io/utils/
train_y = np_utils.to_categorical(y_train, dtype='int32')
test_y = np_utils.to_categorical(y_test, dtype='int32')
#adjusting the shape
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
x_train = x_train.reshape(x_train.shape[0], image_x, image_y, 1)
x_test = x_test.reshape(x_test.shape[0], image_x, image_y, 1)

#explained in Readme file
def keras_model(image_x, image_y):
    num_of_classes = 37
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(5,5), padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #saving the trained data is saved here
    filepath='devanagari1.h5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    #returning the model and the checkpoints of the saved(learned) data
    return model, callbacks_list

model, callbacks_list = keras_model(image_x, image_y)
model.fit(x_train, train_y, validation_data=(x_test, test_y), epochs=10, batch_size=64, callbacks=callbacks_list)
scores = model.evaluate(x_test, test_y, verbose=0)
print("CNN error: %.2f%%" %(100 - scores[1] * 100))
print_summary(model)
model.save('devanagari1.h5')
