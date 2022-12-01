from matplotlib import pyplot
#from keras.datasets import fashion_mnist
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.utils import to_categorical

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

#method for use in colab to save files for group members unable to access veras
def save_data(Xtrain, Ytrain, Xtest, Ytest):
	#scipy.sparse.save_npz("count_sentiment_x_train.npz", count_sentiment_x_train, compressed=True)
	#scipy.sparse.save_npz("count_sentiment_x_test.npz", count_sentiment_x_test, compressed=True)
	np.save("Xtest.npy", Xtest, allow_pickle=False)
	np.save("Xtrain.npy", Xtrain, allow_pickle=False)
	np.save("Ytrain.npy", Ytrain, allow_pickle=False)
	np.save("Ytest.npy", Ytest, allow_pickle=False)

def get_prepped_original_data():
    Xtrain, Ytrain, Xtest, Ytest = load_dataset() # load dataset
    Xtrain, Xtest = prep_pixels(Xtrain, Xtest) # prepare pixel data
    #Debug
    print('Train: X=%s, y=%s' % (Xtrain.shape, Ytrain.shape))
    print('Test: X=%s, y=%s' % (Xtest.shape, Ytest.shape))
    print(Xtrain.shape)
    print(Xtest.shape)
    return Xtrain, Ytrain, Xtest, Ytest

def save_disk_prepped_original_data():
    Xtrain, Ytrain, Xtest, Ytest = load_dataset() # load dataset
    Xtrain, Xtest = prep_pixels(Xtrain, Xtest) # prepare pixel data
    #Debug
    print('Train: X=%s, y=%s' % (Xtrain.shape, Ytrain.shape))
    print('Test: X=%s, y=%s' % (Xtest.shape, Ytest.shape))
    print(Xtrain.shape)
    print(Xtest.shape)

# loads the current version of data (non archived) from the specified folder
def load_local():
		Xtest = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Xtest.npy")
		Xtrain = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Xtrain.npy")
		Ytrain = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Ytrain.npy")
		Ytest = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Ytest.npy")
		print('Train: X=%s, y=%s' % (Xtrain.shape, Ytrain.shape))
		print('Test: X=%s, y=%s' % (Xtest.shape, Ytest.shape))
		print(Xtrain.shape)
		print(Xtest.shape)

		return Xtrain, Ytrain, Xtest, Ytest


#loads versions vectorized into arrays from local directory
def get_prepped_original_data_from_file():
	#Xtest = np.load(file = 'Xtest.npy')
	#Xtrain = np.load(file = 'Xtrain.npy')
	#Ytrain = np.load(file = 'Ytrain.npy')
	#Ytest = np.load(file = 'Ytest.npy')

	Xtest = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Xtest.npy")
	Xtrain = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Xtrain.npy")
	Ytrain = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Ytrain.npy")
	Ytest = np.load(file = r"C:\Users\alici\Dropbox\Class\COMP 551 Applied Machine Learning\Projects\Project 3\MLP\data\Ytest.npy")

	return Xtrain, Ytrain, Xtest, Ytest


