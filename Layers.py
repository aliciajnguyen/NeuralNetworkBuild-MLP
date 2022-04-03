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

        self.dropout_prob = None #by default, no dropout

        #ACTIVATION FUNCTION
        self.activation_function = activation_function_obj.get_func()
        self.activation_function_der = activation_function_obj.get_deriv()

    #input here is X aka z^l-1 (output of the last layer)
    #output is z^l aka this layer's output
    def get_output(self, z):
        #compute activation function
        output = self.activation_function(z) #cross mult correct?
        #print("Hidden layer output:") #debug
        #print(output)
        return output 

    #function to return the derivative of this layer's activation function computed at z(activation)
    def get_af_deriv(self,z):
        dh = self.activation_function_der(z)
        return dh #derivative of h(z) aka pder z/pderiv q sl16

    #to implement dropouts if indicated in fit
    def set_dropout(self, p):
        self.dropout_prob = p
    
    def get_dropout(self):
        return self.dropout_prob
    

        #The * unpacks a tuple into multiple input arguments. 
        # The code is creating a random matrix the same shape as H1 
        # using the shape attribute (which is a tuple) as the dimension inputs to np.random.rand.


#the edge class performs the linear transformations (ie Wv + b)
# note the input layer to first layer DOES NOT have an edge bc IS an edge
#   we'll make sure ever hidden layer is attached to an output edge
#   this is where linear transform happens: Wx + b (after the activation for this layer)
#   and this will be the input to the next layer
#   weight matrix should be W for the last layer
class Edge():
    #n_out will be C if final edge
    def __init__(self, n_in, n_out):       
        #INPUT TO LAYER: h() of last layer
        #OUTPUT:  #self.output = z^l = WX + b: this layer's output later stored 

        #PARAMETERS  (dimensions from constructor of MLP)
        #bias addition handled in constructor, V will be W for final edge
        self.V = np.random.randn(n_in, n_out) #* 0.1   #V dim = M x D
        
    # compute VX + b: z @ V + b, b incorporated into weight and X vector
    #z will be X for first input edge
    def get_output(self, z):
        output = z @ self.V  

        #print("Linear layer output:") #debug
        #print(output)
        return output

    #will be W for the final layer
    def get_params(self):
        return self.V
    
    #for altering initialization
    #will be W for the final layer
    def set_params(self, V):
        self.V = V


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
        #print("Final layer output:")
        #print(output)
        return output 
    
    #cost function passed to constructor
    #categorical cross  entropy cost function
    def get_cost(self, Y, Yh):
        return self.cost_function(Y, Yh)