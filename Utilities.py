import numpy as np

#HIDDEN LAYERS

#activation function (for hidden layers)
def reLu(x):
    return max(0, x)

#TODO tanh and leaky reLu

#FINAL LAYER

#activation function (for final layer) regression 
def identity(x):
    return x

#activation function (for final layer) multiclass classification
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#activation function (for final layer) binary classification
def logistic(x):
    return 1./ (1 + np.exp(-x))





#kind of just rough versions of things being incorporated into the rest of the code:

#logistic = lambda z: 1./ (1 + np.exp(-z))
