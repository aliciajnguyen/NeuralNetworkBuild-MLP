#object for each layer of our MLP
#define a base/parent class that the 3 types of layers will inherit from
#3 different layer articulations after

import numpy as np

#GENERAL CASE
#more detailed explanation in 3 following classes
class Layer:
    
    # Will be overriden to have different constructor for each layer type
    def __init__(self, W, activation_function):       
        pass

    
    #FOR FORWARD PASS

    #input here is X aka z^l-1 (output of the last layer)
    #or X if first layer
    #output is z^l aka this layer's output
    # or Yh if the last layer
    #implement separately in each class
    def get_output(self, input):
        pass 

    #FOR BACKWARD PASS
    #TODO
    def get_params_grad(self, X, output_grad):
        pass

    def get_input_grad(self, Y, output_grad):
        pass


# class for the input layer (called an edge because does special linear trasofrm with V)
# V is special first case, will make our input the size we want according to hidden unit number choice
# only layer that is not attached to the next layer by an edge layer (if we implemnet as edges)
#no activation function, we sent it directly to the first hidden layer
class InputEdge(Layer):
    
    # different constructor than the parent class, differences:
    #   will take V instead of W
    #   no activation function
    def __init__(self, V, b):              
        #INPUT TO LAYER
        # X input ,  dim M x 1 for each instance, for N instances, so D x N
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass as
        #self.input = input 

        #WEIGHT PARAMETERS 
        #will be V for the first layer
        self.V = V

        #BIAS
        #will be the same for the whole layer
        #dimensions according to the weight input passed (same number of cols)
        #initialize to 1 (hyperparameter to tune?)
        #learn later
        #initialize in constructor
        #self.b = b INCORPORATED IN TO WEIGHT VECTOR V

        #self.b = np.ones(V.shape[0]) #WRONG
        
        #ACTIVATION FUNCTION - none for this layer

        #LAYER OUTPUT
        #will be computed in get_output
        #stored in self.output

    #FOR FORWARD PASS

    #   input here is X (original data), dim = N x D
    #   function is must matrix multiplication (@) with V TODO check this logic
    #   output is z^l aka this layer's output
    def get_output(self, X):
        #TODO should we be saving the input here too? prob not just output of last layer
        #TODO is this matrix multiplication correct? check math
        output = (X @ self.V) 
        #output = output + self.b 
        self.output = output
        return output

    #FOR BACKWARD PASS
    #TODO
    def get_params_grad(self, X, output_grad):
        pass

    def get_input_grad(self, Y, output_grad):
        pass

#hidden layer can have whatever relavent activation function
#I'm not yet sure whether to incorporate the linear transformations in this class itsel
#ie not sure if I should do Wx + b IN this layer or if that will make it more complicated for backprop


#   we'll make sure ever hidden layer is attached to an output edge
#   this is where linear transform happens: Wx + b (after the activation for this layer)
#   and this will be the input to the next layer
#   maybe want to make Edge inner class of HiddenLayer if attribute access gets tricky
class HiddenLayer(Layer):

    def __init__(self, W, b, activation_function):       
        #INPUT TO LAYER
        # dim M x 1 (one input from each hidden unit because summation of other components see MLP slide 15)
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass TODO
        #self.input = input 

        #OUTPUT EDGE 
        #we create the output edge, which will hold W and b
        self.out_edge = Edge(self, W, b)
        
        #WEIGHT PARAMETERS AND BIAS in edge associated

        #ACTIVATION FUNCTION
        self.activation_function = activation_function

        #STORED LATER
        #LAYER OUTPUT
        #self.output_hu =  where the output of the activation function stored: activation_function (z)
        #dim 

        #self.edge_output = where output of linear stord (WX + b)
        #dim M x 1
        

    #FOR FORWARD PASS

    #input here is X aka z^l-1 (output of the last layer)
    #output is z^l aka this layer's output
    def get_output(self, z):
        #compute activation function
        output_hu = self.activation_function(z) #cross mult correct?
        

        #need to append an extra col to output_hu for bias
        #so dim will become M + 1 x 1
        #must add bias to X input, so now D + 1 width - an extra col?
        
        bias = np.ones((output_hu.shape[0],1), dtype=float) #append space for each row 
        output_hu = np.append(output_hu, bias, axis=1)

        self.output_hu = output_hu #TODO I assume we save after extra col added

        # compute z @ self.W + self.b
        edge_output = (self.out_edge).get_edge_output(output_hu) # compute z @ self.W + self.b
        self.edge_output = edge_output
        return edge_output #this is what we want as the output to the next activation function

    #FOR BACKWARD PASS
    #TODO
    def get_params_grad(self, X, output_grad):
        pass

    def get_input_grad(self, Y, output_grad):
        pass

#the edge class performs the linear transformations (ie Wv + b)
# note the input layer to first layer DOES NOT have an edge bc IS an edge
class Edge():
    def __init__(self, hidden_layer, W, b):       
        #INPUT TO LAYER
        # dim M x 1 (one input from each hidden unit because summation of other components see MLP slide 15)
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass TODO
        #self.input = input 

        #hidden layer this edge belongs to (is the out edge of)
        self.hidden_layer = hidden_layer

        #PARAMETERS  (initialized in constructor)
        self.W = W

        #BIAS
        #defined in the Edge associated
        #will be the same for the whole layer
        #dimensions according to the weight input passed (same number of cols)
        #initialize to 1 (hyperparameter to tune?)
        #learn later
        #self.b = b       INCORP INTO WEIGHT VECTOR

    # compute WX + b: z @ self.W + self.b
    def get_edge_output(self, z):
        output = z @ self.W 
        #output = output + self.b INCORP INTO WEIGHT VECTOR
        return output



#task is to apply the relavent function for our task
#for this project it will be softmax for multiclass classification
#I kept it generalizable for future exapnsion
#though note to do so we'd have to change the cost function (and how it's computed)
class OutputLayer(Layer):

    def __init__(self, activation_function):       
        #INPUT TO LAYER
        # dim M x 1
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass TODO
        #self.input = input 

        #ACTIVATION FUNCTION
        self.activation_function = activation_function

        #LAYER OUTPUT
        #will be computed in get_output
        #stored in self.output = yh
        #dim 1 Cx D?? 

    #FOR FORWARD PASS

    #input here is X aka z^l-1 (output of the last layer)
    #output is yh
    #here it will be softmax (multiclass), logistic (binary class) or identity (regression)
    # each Y is dim C x 1
    # so each Yh should be dim C x 1 ?
    def get_output(self, z):
        output = self.activation_function(z) 
        self.output = output
        return output 

    #FOR BACKWARD PASS
    #TODO

    def get_cost(self):
        pass

    def get_params_grad(self, X, output_grad):
        pass

    def get_input_grad(self, Y, output_grad):
        pass
