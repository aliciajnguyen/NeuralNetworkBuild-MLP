#fit and predict go here
#import Utilities as u

import numpy as np
#%matplotlib notebook
#%matplotlib inline
import matplotlib.pyplot as plt
#from IPython.core.debugger import set_trace
import warnings
warnings.filterwarnings('ignore')

logistic = lambda z: 1./ (1 + np.exp(-z))

class MLP:
    
    def __init__(self, M = 64):
        self.M = M
            
    def fit(self, x, y, optimizer): #optimizer is GD in our case
        N,D = x.shape
        def gradient(x, y, params): #computes grad of loss wrt dparams ?
            v, w = params
            z = logistic(np.dot(x, v)) #N x M #forward pass
            yh = logistic(np.dot(z, w))#N #forward pass
            dy = yh - y #N #using L2 loss even tho doing class? shouldn't, right way is to use cross entropy loss
            #dy is L / L _ y but we use dy?  dy = dL/dy
            dw = np.dot(z.T, dy)/N #M #take grad step in backwards direction, dw= dy/dw
            dz = np.outer(dy, w) #N x M #compute 
            dv = np.dot(x.T, dz * z * (1 - z))/N #D x M 
            dparams = [dv, dw] #store gradients of two parametsrs in a list
            return dparams
        
        w = np.random.randn(self.M) * .01 #out of func intiialize weights and store initial versions
        v = np.random.randn(D,self.M) * .01
        params0 = [v,w]
        self.params = optimizer.run(gradient, x, y, params0) #pass grad , x, ,y, initial params 
        #returns optimized parameters
        return self
    
    def predict(self, x): #just a function
        v, w = self.params
        z = logistic(np.dot(x, v)) #N x M
        yh = logistic(np.dot(z, w))#N
        return yh