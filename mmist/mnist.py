# mnist classification
# backend: Theano (make sure to disable fastmath i.e. nvcc.fastmath=false. if you want to add many epoches and wish to avoid the nan issue)
# you can play with the layers and maybe add a few (maybe dropout layers).

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# To make the results reproducible
seed = 5
np.random.seed(seed)

# Load and prepare dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=1);
x_test = np.expand_dims(x_test, axis=1);
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Define the model
def get_model():
	model = Sequential()
	model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', init='normal', input_shape=(1,28,28)))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='normal'))
	model.add(Flatten())
	model.add(Dense(128, init='normal', activation='relu'))
	model.add(Dense(32, init='normal', activation='relu'))
	model.add(Dense(10, init='normal', activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = get_model()
history = model.fit(x_train, y_train, batch_size=10, nb_epoch=4, shuffle=True, verbose=1,  validation_data=(x_test, y_test))


