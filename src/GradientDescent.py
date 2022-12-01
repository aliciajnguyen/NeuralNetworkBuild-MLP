#https://colab.research.google.com/drive/1VEYvm4Z2denIM2Gy1j9K7tqmA38zwKaQ#scrollTo=fa3GgubSfY2R
#TODO - did stochastic gradient descent ever even get implemented?
import numpy as np

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
                params[p] -= self.learning_rate * grad[p]  #DIM PROB HERE

            t += 1
            norms = np.array([np.linalg.norm(g) for g in grad])
        return params