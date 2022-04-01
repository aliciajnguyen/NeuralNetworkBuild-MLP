#LEARNING INTERFACES/ abstract methods in python # https://realpython.com/python-interface/#using-abstract-method-declaration
import numpy as np

#HIDDEN LAYERS

#parent class - defines functions all children have (to get the function and it's derivative)
class ActivationFunction:
    def get_func(self):
        return self.function
    def get_deriv(self):
        return self.derivative
    #to initialized parameter matrix based on activation type, matrix already initialized to random
    def param_init_by_activ_type(self, V, M_last):
        return self.param_initializer(V, M_last)

#activation function (for hidden layer) 
class ReLU(ActivationFunction):
    def __init__(self):
        reLU = lambda x: np.maximum(0, x)
        self.function = reLU
        #reLU_der = lambda x: 0 if x < 0 else 1 
        reLU_der = lambda x: (x > 0) *1  #PROBLEM, should be elementwise


        #for np arrays (bc boolean expressions involving them turned into arrays of values of these expre for els in said array) 
        self.derivative = reLU_der
        #custom weight initialization based on af, V will be rand already
        self.param_initializer = lambda V, M_last : V * np.sqrt(2/M_last) #He init for ReLu

#activation function (for hidden layer) 
class tanh(ActivationFunction):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        self.function = tanh
        tanh_der = lambda x: 1 - np.tanh(x) * np.tanh(x) #TODO check elementwise here
        self.derivative = tanh_der
        #custom weight initialization based on af, V will be rand already
        self.param_initializer = lambda V, M_last : V * np.sqrt(1/M_last) #Xavier init for tanh


#activation function (for hidden layer)
#lambda is by default set to 0.2, but can be changed as HP
class LeakyReLU(ActivationFunction):
    def __init__(self, lam=0.2):
        self.function = self.leaky_ReLU
        self.derivative = self.der_leaky_ReLU
        #custom weight initialization based on af, V will be rand already
        self.param_initializer = lambda V, M_last : V * np.sqrt(2/M_last) #He init for ReLu
        
    #sum compacted, bc always > 0 with min addition
    def leaky_ReLU(x, lam):                           
        return np.where(x > 0, x, x * lam)                          

    def der_leaky_ReLU(x, lam):
        data = [1 if value>0 else lam for value in x]
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


#    def leaky_ReLU(x):
#        #data = [max(0.05*value,value) for value in x]
#        #return np.array(data, dtype=float)

#    def leaky_ReLU(x):
#
#        # first approach                           
#        leaky_way1 = np.where(x > 0, x, x * 0.01)                          #
#
#        # second approach ()                                                                  
#        y1 = ((x > 0) * x)                                                 
#        y2 = ((x <= 0) * x * 0.01)                                         
#        leaky_way2 = y1 + y2  

