#object for each layer of our MLP
#define a base/parent class that the 3 types of layers will inherit from
#3 different layer articulations after, 1 "edge" layer articulation

import numpy as np

#GENERAL CASE
class Layer:
    
    # Will be overriden to have different constructor for each layer type
    def __init__(self):
        #self.activations each layer must have       
        pass
    
    #FOR FORWARD PASS
    def get_output(self, input):
        pass 

    #FOR BACKWARD PASS
    def get_input_grad(self, Y, output_grad):
        pass


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
        

    #FOR FORWARD PASS

    #   input here is X (original data), dim = N x D
    #   function is must matrix multiplication (@) with V 
    #   output is z^l aka this layer's output
    def get_output(self, X):
        #TODO should we be saving the input here too? prob not just output of last layer
        #TODO is this matrix multiplication correct? check math
        output = (X @ self.V) 
        self.output = output #INPUT LAYER
        return output

    #FOR BACKWARD PASS
    #TODO
    def get_params_grad(self, X, output_grad):
        pass

    def get_input_grad(self, Y, output_grad):
        pass

#hidden layer can have whatever relavent activation function


#   we'll make sure ever hidden layer is attached to an output edge
#   this is where linear transform happens: Wx + b (after the activation for this layer)
#   and this will be the input to the next layer
class HiddenLayer(Layer):

    def __init__(self, activation_function):       
        #INPUT TO LAYER
        # computed from last edge layer
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass TODO

        #ACTIVATION FUNCTION
        self.activation_function = activation_function

        #STORED LATER
        #LAYER OUTPUT
        #self.output=  where the output of the activation function stored: activation_function (z)

    #FOR FORWARD PASS

    #input here is X aka z^l-1 (output of the last layer)
    #output is z^l aka this layer's output
    def get_output(self, z):
        #compute activation function
        output = self.activation_function(z) #cross mult correct?
        
        #need to append an extra col to output_hu for bias
        #so dim will become M + 1 x 1
        #must add bias to X input, so now D + 1 width - an extra col?
        bias = np.ones((output.shape[0],1), dtype=float) #append space for each row 
        output = np.append(output, bias, axis=1)

        self.output = output #HID LAYER
        print("Hidden layer output:")
        print(output)

        return output #this is what we want as the output to the next activation function

    #FOR BACKWARD PASS
    #TODO
    def get_params_grad(self, X, output_grad):
        pass

    def get_input_grad(self, Y, output_grad):
        pass

#the edge class performs the linear transformations (ie Wv + b)
# note the input layer to first layer DOES NOT have an edge bc IS an edge
#   we'll make sure ever hidden layer is attached to an output edge
#   this is where linear transform happens: Wx + b (after the activation for this layer)
#   and this will be the input to the next layer
#   weight matrix should be W for the last layer
class Edge(Layer):
    def __init__(self, V):       
        #INPUT TO LAYER
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass TODO
     
        #PARAMETERS  (initialized in constructor of MLP)
        self.V = V

        #LATER CALCULATED

        #self.output = where output of linear stord (WX + b)
        #dim M x 1

        #BIAS
        #defined in the Edge associated
        #will be the same for the whole layer
        #dimensions according to the weight input passed (same number of cols)
        #initialize to 1 (hyperparameter to tune?)
        #learn later
        #self.b = b       INCORP INTO WEIGHT VECTOR

    # compute WX + b: z @ self.W + self.b
    # b INCORP INTO WEIGHT VECTOR
    def get_output(self, z):
        output = z @ self.V  # compute z @ self.W + self.b
        self.output = output #EDGE
        return output

    #FOR BACKWARD PASS
    #TODO
    def get_params_grad(self, X, output_grad):
        pass

    #gradient at the input of this layer
    def get_input_grad(self, Y, output_grad):
        pass
        #return output_grad @ self.W.Y


#layer task is to apply the relavent function for our task
#for this project it will be softmax for multiclass classification
#I kept it generalizable for future exapnsion
#though note to do so we'd have to change the cost function (and how it's computed)
class OutputLayer(Layer):

    def __init__(self, activation_function):       
        #INPUT TO LAYER
        #input here is X aka z^l-1 (output of the last layer)
        #calculated in first pass TODO

        #ACTIVATION FUNCTION
        self.activation_function = activation_function

        #LAYER OUTPUT
        #will be computed in get_output
        #dim ? determined by the weight shape of the last yayer
        #stored in self.output 

    #FOR FORWARD PASS

    #input here is X aka z^l-1 (output of the last layer)
    #output is yh
    #here it will be softmax (multiclass), logistic (binary class) or identity (regression)
    # each Y is dim C x D?
    def get_output(self, z):
        output = self.activation_function(z) 
        self.output = output
        print("Final layer output:")
        print(output)
        return output  #OUTPUT LAYER

    #FOR BACKWARD PASS
    #TODO

    #NOTE that to expand this beyond softmax multiclsas we'd have to update cost
    #categorical cross  entropy cost function
    def get_cost(self, Y, Yh):
        return - np.multiply(Yh, np.log(Y)).sum() / Y.shape[0]

    def get_input_grad(self, Y, output_grad):
        return (Y - T) / Y.shape[0]