from matplotlib import pyplot
from keras.datasets import fashion_mnist
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	#reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 784))
	testX = testX.reshape((testX.shape[0], 784))
	#trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	#testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

    #plotting 
#for i in range(9):  
  #pyplot.subplot(330 + 1 + i)
  #pyplot.imshow(trainX[i+100], cmap=pyplot.get_cmap('hot'))
#pyplot.show()


# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm