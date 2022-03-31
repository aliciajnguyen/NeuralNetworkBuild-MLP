#object for each layer of our MLP
#define a base/parent class that the 3 types of layers will inherit from
#3 different layer articulations after, 1 "edge" layer articulation

import itertools
import numpy as np
import ActivationFunctions as af

#GENERAL STRUCTURE
#Parent class, use for documenting other layers - necessary?
class Layer:
    # Will be overriden to have different constructor for each layer type
     #FOR FORWARD PASS
    def get_output(self, input):
        pass 
    #FOR BACKWARD PASS
    def get_input_grad(self, Y, output_grad):
        pass
    #all classes inherit:
    def get_activations():
        return self.output


# class for the input layer (called an edge because does special linear trasofrm with V)
# V is special first case, will make our input the size we want according to hidden unit number choice
# only layer that is not attached to the next layer by an edge layer (is an edge)
# no activation function, we sent it directly to the first hidden layer
class InputEdge(Layer):
    
    # different constructor than the parent class, differences:
    #   will take V instead of W
    #   no activation function
    def __init__(self, V):              
        #INPUT TO LAYER: X input ,  dim M x 1 for each instance, for N instances, so D x N
        #OUTPUT OF LAYER
        #last saved as self.output in get_output

        self.V = V         #WEIGHT PARAMETERS 

    #   input here is X (original data), dim = N x D
    #   function is must matrix multiplication (@) with V 
    #   output is z^l aka this layer's output
    def get_output(self, X):
        output = (X @ self.V) 
        self.output = output #INPUT LAYER
        return output

    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(
            np.nditer(self.W, op_flags=['readwrite']),
            np.nditer(self.b, op_flags=['readwrite']))
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T @ output_grad
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(
            np.nditer(JW), np.nditer(Jb))]
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad @ self.V.T

#hidden layer can have whatever relavent activation function


#   we'll make sure ever hidden layer is attached to an output edge
#   this is where linear transform happens: Wx + b (after the activation for this layer)
#   and this will be the input to the next layer
class HiddenLayer(Layer):

    def __init__(self, activation_function_obj):       
        #INPUT TO LAYER
        # computed from last edge layer
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass
        #LAYER OUTPUT
        #self.output=  where the output of the activation function stored: activation_function (z)

        #ACTIVATION FUNCTION
        self.activation_function = activation_function_obj.get_func()
        self.activation_function_der = activation_function_obj.get_deriv()

    #input here is X aka z^l-1 (output of the last layer)
    #output is z^l aka this layer's output
    def get_output(self, z):
        #compute activation function
        output = self.activation_function(z) #cross mult correct?
        
        #need to append an extra col to output for bias
        #so dim will become ((M + 1) x N)
        bias = np.ones((output.shape[0],1), dtype=float)
        output = np.append(output, bias, axis=1)

        self.output = output

        print("Hidden layer output:") #debug
        print(output)

        return output 

    #get gradient of inputs at this layer
    #pass the derivative of whatever activation function we're using
    def get_input_grad(self, Y, output_grad):
        

        return np.multiply(self.activation_function_der(Y), output_grad)

#the edge class performs the linear transformations (ie Wv + b)
# note the input layer to first layer DOES NOT have an edge bc IS an edge
#   we'll make sure ever hidden layer is attached to an output edge
#   this is where linear transform happens: Wx + b (after the activation for this layer)
#   and this will be the input to the next layer
#   weight matrix should be W for the last layer
class Edge(Layer):
    def __init__(self, V, isFinalEdge=False):       
        #INPUT TO LAYER
        #input here is X aka z^l-1 (output of the last layer)
        #OUTPUT
        #self.output = where output of linear stord (WX + b)

        #PARAMETERS  (initialized in constructor of MLP)
        self.V = V

    # compute WX + b: z @ self.W + self.b
    def get_output(self, z):
        output = z @ self.V  # compute z @ self.W + self.b, b INCORP INTO WEIGHT VECTOR
        self.output = output #EDGE
        return output

    def get_params_grad(self, X, output_grad):
        pass

    #gradient at the input of this layer
    def get_input_grad(self, Y, output_grad):
        pass
        #return output_grad @ self.W.Y
        
    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(
            np.nditer(self.W, op_flags=['readwrite']),
            np.nditer(self.b, op_flags=['readwrite']))

    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T @ output_grad
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(
            np.nditer(JW), np.nditer(Jb))]
    
    #gradients of inputs at this layer
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        #return output_grad @ self.V.Y
        return output_grad @ self.V.T


#layer task is to apply the relavent function for our task
#for this project it will be softmax for multiclass classification
#I kept it generalizable for future exapnsion
#though note to do so we'd have to change the cost function (and how it's computed)
class OutputLayer(Layer):

    def __init__(self, activation_function_obj, cost_function):       
        self.activation_function = activation_function_obj.get_func()  #ACTIVATION FUNCTION
        self.cost_function = cost_function              #COST FUNCTION
        #INPUT TO LAYER
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass
        #LAYER OUTPUT
        #will be computed in get_output, dim ? determined by the shape of weight matrix in last layer
        #stored in self.output , here Yh

    #input here is X aka z^l-1 (output of the last layer), output is yh
    #here it will be softmax (multiclass), logistic (binary class) or identity (regression)
    # each Yh is dim C x N?
    def get_output(self, z):
        output = self.activation_function(z) 
        self.output = output
        print("Final layer output:")
        print(output)
        return output 

    #cost function passed to constructor
    #categorical cross  entropy cost function
    def get_cost(self, Y, Yh):
        return self.cost_function(Y, Yh)

    def get_input_grad(self, Y, Yh):
        return (Y - Yh) / Y.shape[0]
    #    return Yh - Y  #should be dy