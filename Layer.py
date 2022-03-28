#object for each layer of our MLP

from tkinter.tix import Tree
import numpy as np

#maybe want decorator DP for first and last layers?
class Layer:

    #GENERAL CASE
    def __init__(self, W, activation_function, isFirstLayer=False, isLastLayer=False):
        #don't even think this will be necessary
        self.isFirstLayer = isFirstLayer
        self.isLastLayer = isLastLayer
        
        #INPUT TO LAYER
        # dim M x 1
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass TODO
        #self.input = input 

        #WEIGHT PARAMETERS 
        #will be v for the first layer
        self.W = W

        #BIAS
        #will be the same for the whole layer
        #dimensions according to the weight input passed (same number of cols)
        #initialize to 1 (hyperparameter to tune?)
        #learn later
        self.bias = np.ones(W.shape[0])

        #ACTIVATION FUNCTION
        self.activation_function = activation_function

        #LAYER OUTPUT
        #will be computed in get_output
        #stored in self.output

    #FOR FORWARD PASS

    #input here is X aka z^l-1 (output of the last layer)
    #output is z^l aka this layer's output
    def get_output(self, input):
        output = self.activation_function(np.dot(X, self.W) + self.b)
        self.output = output
        return output

    #FOR BACKWARD PASS
    #TODO
    def get_params_grad(self, X, output_grad):
        pass

    def get_input_grad(self, Y, output_grad):
        pass