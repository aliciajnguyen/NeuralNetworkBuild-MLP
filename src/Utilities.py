import numpy as np

#COST FUNCTIONS
def cat_cross_entropy(Y, Yh):
    return - np.multiply(Yh, np.log(Y)).sum() / Y.shape[0]

#vectorized version for softmax + one hot encoding
def evaluate_acc(y, yh):
    accuracy = np.sum(np.all(y == yh, axis=1))/y.shape[0]
    print("-------------------------------------")
    print(f'test accuracy: {accuracy}')
    print("-------------------------------------")
    return accuracy

#np.sum(np.all(a == b, axis=1))

#NON VECTORIZED VERSION
def evaluate_accuracy(y, yh):
    accuracy = np.sum(yh == y)/y.shape[0]
    print(f'test accuracy: {accuracy}')
    return accuracy

#HIDDEN LAYERS

#activation function (for hidden layers)
#TODO make sure this is getting applied elementwise!
def reLu(x):
    #return max(0, x)
    return np.maximum(0, x)

#TODO tanh and leaky reLu

#FINAL LAYER

#activation function (for final layer) regression 
def identity(x):
    return x

#activation function (for final layer) multiclass classification
def softmax(x):
    #caution doesn't apply on correct axis
    #e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum()
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

#activation function (for final layer) binary classification
def logistic(x):
    return 1./ (1. + np.exp(-x))

#pre-vectorized version of cat_cross_entropy
# - (T * np.log(Y)).sum()

#kind of just rough versions of things being incorporated into the rest of the code:

#logistic = lambda z: 1./ (1 + np.exp(-z))