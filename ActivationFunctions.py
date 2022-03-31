#LEARNING INTERFACES/ abstract methods in python
# https://realpython.com/python-interface/#using-abstract-method-declaration
##create base (parent) clsas 

import numpy as np

#HIDDEN LAYERS

#parent class - defines functions all children have (to get the function and it's derivative)
class ActivationFunction:
    def get_func(self):
        return self.function
    def get_deriv(self):
        return self.derivative

#activation function (for hidden layer) 
class ReLU(ActivationFunction):
    def __init__(self):
        reLU = lambda x: np.maximum(0, x)
        self.function = reLU
        reLU_der = lambda x: 0 if x < 0 else 1 
        self.derivative = reLU_der

#activation function (for hidden layer) 
class tanh(ActivationFunction):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        self.function = tanh
        tanh_der = lambda x: 1 - np.tanh(x) * np.tanh(x)
        self.derivative = tanh_der

#activation function (for hidden layer)
# TODO parameterize 
class LeakyReLU(ActivationFunction):
    def __init__(self):
        self.function = self.leaky_ReLU
        self.derivative = self.der_leaky_ReLU

    def leaky_ReLU(x):
        data = [max(0.05*value,value) for value in x]
        return np.array(data, dtype=float)

    def der_leaky_ReLU(x):
        data = [1 if value>0 else 0.05 for value in x]
        return np.array(data, dtype=float)


#FINAL LAYER: #don't bother with derivatives

#activation function (for final layer) multiclass classification
class SoftMax(ActivationFunction):
    def __init__(self):
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.function = softmax
        #derivative more complicated than it looks, calc jacobian differnetly
       
#activation function (for final layer) regression
class identity(ActivationFunction):
    def __init__(self):
        identity = lambda x: x
        self.function = identity

#activation function (for final or hidden layer) binary classification
class logistic(ActivationFunction):
    def __init__(self):
        logistic = lambda x: 1./ (1. + np.exp(-x))
        self.function = logistic
        logistic_der = lambda x: np.multiply(x, (1 - x))
        self.derivative = logistic_der
