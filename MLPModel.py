import collections
import numpy as np
import GradientDescent as gd
import Utilities as u
import Layers as l
import GradientDescent as gd
import ActivationFunctions as af

np.random.seed(1234)

class MLP:
    
    #currently the parameter initialization is hardcoded to be random, but could be changed later
    def __init__(self, M, D, C, hidden_activation_func_list, output_activation_func, cost_function = u.cat_cross_entropy, parameter_init_type = "RANDOM"):
        self.M = M     # M =number of hidden units in hidden layers (width)
        self.C = C     # C outputs (number of classes)        
        self.D = D     # D inputs (number of x inputs) CONFUSION WITH N
       
        #W dim = C X M
        #V dim = M x D  - np weird ordering in matrix creation careful
        #self.V, self.W = self.initialize_parameters(parameter_init_type) MOVED TO CREATE LAYERS
        
        #list of represent all layers in mlp
        self.layers = self.create_layers(hidden_activation_func_list, output_activation_func, cost_function)
                 
        self.activations = [] #a list of activations
        # computed for every activation function layer and linear transformation layer (edge)

        #OBJECT ATTRIBUTES LATER COMPUTED
        #num hidden_layers just len of activation_func_list
        #self.learned_params  #what we learn using gradient descent in fit function IF WE DON'T DO STOCHASTIC
        #self.N  = the number of training instances fit to
            
    #initialize the weights in the constructor according to intialization type (parameter to tune)
    #note if that hidden activation func list is empty, no hidden layers constructed
    #TODO delete this method
    def initialize_parameters(self, init_type):
        if init_type == "RANDOM":
            #note extra col for BIAS 
            V = np.random.randn(self.D+1,self.M) * .01
            W = np.random.randn(self.M+1, self.C) * .01 
            #I think biases are actually just incorporated as extra dim in X and W 
        #elif init_type == "ZERO":
           #put other paramater init here
        return V, W 

    #create layer list here for model
    #called in class intializer
    #if there are not hidden layers I thiiink this should work, TODO test for no HL case (Task 3.1) I think just log regr
    #returns a list of all layers
    #hard coded so that all layers have to have the same number hidden units but this could be changed
    def create_layers(self, hidden_activation_func_list, output_activation_func, cost_function):
        layers_list = []    #list of all layers

        #dimensions for parameter matrices with bias additions (one's col added to X too)
        Dplusbias = self.D +1       #V dim = (D+1, M)
        Mplusbias = self.M +1       #W dim = (M+1, C)

        #account for case with no hidden layers (log regression)
        if len(hidden_activation_func_list) == 0 or hidden_activation_func_list == None:
            layers_list.append(l.FinalEdge(self.D, self.C))       #create first edge (from X to first HU) and add to list
            #no bias this case
        else:
            layers_list.append(l.Edge(Dplusbias, self.M))       #create first edge (from X to first HU) and add to list
    
            #create hidden layers: length of passed activation funcs determines numbre of hidden layers
            last_i = len(hidden_activation_func_list)-1
            for i, activation_function in enumerate(hidden_activation_func_list):             
                hid_layer = l.HiddenLayer(activation_function) 
                layers_list.append(hid_layer)

                #special case for the edges before the final output layer weight matrix W instead of V
                edge = l.Edge(Dplusbias, self.M) if i != last_i else l.FinalEdge(Mplusbias, self.C)
                layers_list.append(edge)

        layers_list.append(l.OutputLayer(output_activation_func, cost_function))         #create output layer
        return layers_list

    #Compute forward pass, outputs stored in layer object as needed
    def forward_pass(self, X):
        #scope in python is a function!
        last_index = len(self.layers)-1
        for i,layer in enumerate(self.layers):
            input = X if i == 0 else z # X will be first layer input, otherwise it's the output, z, of last layer
            #I don't actually think the next line could ahppen caution
            #if i != last_index: self.activations.append(input) #possible bug appending X but I don't think it'll matter
            self.activations.append(input) #possible bug appending X but I don't think it'll matter
            z = layer.get_output(input) #TODO - does the final layer get appended?
        yh = z # last value computed is yh, rename for consistency
        return yh
    
    #perform the backward pass
#        return list(parameter_gradients)  # Return the parameter gradients
    def backward_pass(self, X, Y, Yh):
        # Yh = N X C
        
        #goal
        #v_params = collections.deque()
        #w_params = collections.deque()

        layers = self.layers   # edges and activation layers
        activations = self.activations #output of each layer
        
# Hiden Layer
#def get_grad(dy, V):
#        dz = np.dot(dy, V.T)
#        return dz

#EDGE     #z will be x at first layer?
#    def get_params(self, z, dz, N):
#        dv = np.dot(z.T, dz * z * (1 - z))/N #D x M 

        #last layer and edge special case:
        #compute dy = pderiv L wrt W
        final_layer = layers.pop()              #get output layer
        dy = final_layer.get_grad(self, Y, Yh)  
        z = activations.pop()                   #need last activations (to compute pderiv for input from below)
        final_edge = layers.pop()               #get last edge with W
        dw = final_edge.get_params(z, dy, self.N)  

        w_params = dw       # save for optimization layer
        dv = dw             #for iterating in the following loop

        #reverse the layers (propograte from back)
        for layer in reversed(layers):   
            
            if isinstance(layer, l.HiddenLayer):
               z = activations.pop()  # Get the activations of the last layer on the stack
               dz = layer.get_grad(dy, W)        #WILL NEED TO UPDATE COST, also SUBSET of V?
            #TODO: W in this line should be the weights from the last layer!

            else: # will be an edge
                dv = layer.get_params (z, dz, self.N)

        v_params = list(v_params)
        return v_params, w_params


    def fit(self, X, Y, learn_rate=1, gd_iterations=100, dropout_p=0):
        N,D = X.shape 
 
        #bias implementation: ADD COLS of 1 to x, (must stay 1 to relect weight value)
        bias = np.ones((N,1), dtype=float)
        X = np.append(X, bias, axis=1)

        #DEBUG
        Yh = self.forward_pass(X)     
        #learned_params = self.backward_pass(X, Y, Yh)
        
        #def gradient(X, Y, params):    
        #    Yh = self.forward_pass(X)     
        #    learned_params = self.backward_pass(X, Y, Yh)
        #    return learned_params
        
        #get list of params learned from back prop
        #params0 = [v,w] 

        #create GradientDescent obj here and pass it our HP's
        #take out of the method pass tf
        #optimizer = gd.GradientDescent(learning_rate=learn_rate, max_iters=gd_iterations)

        #actually run GD
        #optimizer here is gd, passed to the fit function
        #self.learned_params = optimizer.run(gradient, x, y, params0) #pass grad , x, ,y, initial params 
        
        #decide how to store parameters for fit
        return self
    
    def predict(self, X): 
        Vlearned, Wlearned = self.learned_params
        yh = self.forward_pass(X , Vlearned, Wlearned)

        #softmax returns probs so much use argmax
        #Yh = np.argmax(Yh, axis = 1)
        return yh #should be dim N