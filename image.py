	
import numpy as np
import os
import cv2
import pandas as pd

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.utils import np_utils


image_height = 480
image_width = 640
nb_classes = 10
batch_size = 1
nb_epoch = 5

def load_train():
    X_train = []
    y_train = []
    heights = pd.read_csv('heights.csv')
    print('Read train images')
    for index, row in heights.iterrows():
        image_path = os.path.join('images', 'train', str(int(row['img'])) + '.jpg')
        img = cv2.resize(cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR), (image_height, image_width) ).astype(np.float32)
        img = img.transpose((2,0,1))
        X_train.append(img)
        y_train.append( [ row['height'] ] )
    #print('size of xtrain ',X_train, 'size of Y ', y_train )
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    return X_train, Y_train



def create_model():
    nb_filters = 64
    nb_conv = 2

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='same',
                            input_shape=( 3, image_width, image_height) ) )
    print(model.output_shape, " OUTPUT")
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    return model

def train_model(X_train, Y_train):
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
	model = create_model()
	X_train = np.array(X_train)
	X_valid = np.array(X_valid)
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_valid, y_valid) )
	score = model.evaluate(X_valid, y_valid, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
X_train, Y_train = load_train()
train_model(X_train, Y_train)
