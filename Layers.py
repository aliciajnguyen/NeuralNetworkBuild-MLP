#object for each layer of our MLP
#define a base/parent class that the 3 types of layers will inherit from
#3 different layer articulations after, 1 "edge" layer articulation

import numpy as np
import ActivationFunctions as af
np.random.seed(1)

#hidden layer can have whatever relavent activation function
#   we'll make sure ever hidden layer is attached to an output edge
#   where h() happens
# conncet to edgess
class HiddenLayer():
    def __init__(self, activation_function_obj):       
        #INPUT TO LAYER: computed from last edge = Z aka z^l-1 = WX + b
        #LAYER OUTPUT: h()
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

    #function to return the derivative of this layer's activation function computed at z(activation)
    def get_af_deriv(self,z):
        dh = self.activation_function_der(z)
        return dh #derivative of h(z) aka pder z/pderiv q sl16

#the edge class performs the linear transformations (ie Wv + b)
# note the input layer to first layer DOES NOT have an edge bc IS an edge
#   we'll make sure ever hidden layer is attached to an output edge
#   this is where linear transform happens: Wx + b (after the activation for this layer)
#   and this will be the input to the next layer
#   weight matrix should be W for the last layer
class Edge():

    def __init__(self, hu_num_in, hu_num_out, activation_func_after=None):       
        #INPUT TO LAYER: h() of last layer
        #OUTPUT:  #self.output = z^l = WX + b: this layer's output later stored 

        #PARAMETERS  (dimensions from constructor of MLP)
        #bias addition handled in constructor
        self.V = np.random.randn(hu_num_in, hu_num_out) * 0.1
        
        #later we'll ensure our weights are initialized properly given what kind of activation we have
        #which is why we link the activation to it's previous edge
        self.af_after = activation_func_after
        #TODO

    # compute VX + b: z @ V + b, b incorporated into weight and X vector
    #z will be X for first input edge
    def get_output(self, z):
        output = z @ self.V  
        self.output = output 
        print("Linear layer output:") #debug
        print(output)
        return output

    def get_params(self):
        return self.V

#special edge before final layer because weight parameters have different dimensions
class FinalEdge():
    def __init__(self, hu_num_in, C):

        #PARAMETERS  (dimensions from constructor of MLP)
        #number of HU's in last layer and num of classes, bias addition handled in constructor
        self.W = np.random.randn(hu_num_in, C) * 0.1

    # compute WX + b: z @ W + b, b incorporated into weight and X vector
    def get_output(self, z):
        output = z @ self.W  
        self.output = output 
        print("Final Linear layer output:")
        print(output)
        return output
    
    def get_params(self):
        return self.W

#layer task is to apply the relavent function for our task
#for this project it will be softmax for multiclass classification
#I kept it generalizable for future exapnsion
#though note to do so we'd have to change the cost function as passed to the MLP constructor
class OutputLayer():

    def __init__(self, activation_function_obj, cost_function):       
        self.activation_function = activation_function_obj.get_func()  #ACTIVATION FUNCTION
        self.cost_function = cost_function              #COST FUNCTION
        #INPUT TO LAYER: x z^l-1 =  WX+ b
        #LAYER OUTPUT: yh (though note need argmax on to predict on)
        #stored in self.output

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