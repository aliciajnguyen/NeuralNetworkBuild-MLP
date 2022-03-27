#https://colab.research.google.com/drive/1VEYvm4Z2denIM2Gy1j9K7tqmA38zwKaQ#scrollTo=fa3GgubSfY2R
import numpy as np
#%matplotlib notebook
#%matplotlib inline
#import matplotlib.pyplot as plt
#from IPython.core.debugger import set_trace
import warnings
warnings.filterwarnings('ignore')


class GradientDescent:
    
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        
    def run(self, gradient_fn, x, y, params):
        norms = np.array([np.inf])
        t = 1
        while np.any(norms > self.epsilon) and t < self.max_iters:
            grad = gradient_fn(x, y, params)
            for p in range(len(params)):
                params[p] -= self.learning_rate * grad[p]
            t += 1
            norms = np.array([np.linalg.norm(g) for g in grad])
        return params

""" not mlp version

#just the gradient eqn
def gradient(x, y, w):                          # define the gradient function
    yh =  x @ w  #@ used as binary operator for matrix multiplication x in terms of w_t?
    N, D = x.shape
    grad = .5*np.dot(yh - y, x)/N #dot product #/N because scaling?? thing that matters : diff (yhat - y ) * x
    return grad

class GradientDescent:
    
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8, record_history=False):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.record_history = record_history
        self.epsilon = epsilon
        if record_history:
            self.w_history = []                 #to store the weight history for visualization
            
    def run(self, gradient_fn, x, y, w):
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(x, y, w)               # compute the gradient with present weight
            w = w - self.learning_rate * grad         # weight update step
            if self.record_history:
                self.w_history.append(w)
            t += 1
        return w

"""