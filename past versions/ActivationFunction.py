#LEARNING INTERFACES/ abstract methods in python
# https://realpython.com/python-interface/#using-abstract-method-declaration
##create base (parent) clsas - or do we want interface? #

import numpy as np

#parent class - rather than interface bc implement here?
class ActivationFunction:
    def get_func(self):
        return self.function
    def get_deriv(self):
        return self.derivative

#activation function (for hidden layer) 
class ReLu(ActivationFunction):
    def __init__(self):
        reLu = lambda x: max(0, x)
        self.function = reLu
        reLu_der = lambda x: 0 if x < 0 else 1 
        self.derivative = reLu_der

#activation function (for final layer) multiclass classification
class SoftMax(ActivationFunction):
    def __init__(self, lname):
        self.function = self.softmax
        #more complicated than it looks
        #softmax_der = lambda x: 
        #self.derivative = softmax_der
    def softmax(self, x):
        #Compute softmax values for each sets of scores in x
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()





## implement forall activation funcs desired (activation functions in utitilies now)
