import numpy as np
import GradientDescent as gd
import Utilities as u
import Layers as l
np.random.seed(1234)


class MLP:
    
    #should take as input the activ func (eg Relu)
    #numbre of hidden layers (eg 2)
    #numbre of hidden units in hidden layers
    #initialize weights and biases
    #also def loss function here depending on selection of output layer
    #note if that hidden activation func list is empty, no hidden layers constructed
    def __init__(self, M, D, C, hidden_activation_func_list, output_activation_func, parameter_init_type = "RANDOM"):
        self.M = M     # M =number of hidden units in hidden layers (width)
        self.C = C     # C outputs (number of classes)        
        self.D = D     # D inputs (number of x inputs) CONFUSION WITH N
       

        #W dim = C X M
        #V dim = M x D  - np weird ordering in matrix creation careful
        #self.V, self.W, self.bw, self.blast = self.initialize_parameters(parameter_init_type)
        self.V, self.W = self.initialize_parameters(parameter_init_type)


        #list of represent all layers in mlp
        self.layers = self.create_layers(hidden_activation_func_list, output_activation_func)
                 
        self.activations = [] #a list of activations
        # computed for every activation function layer and linear transformation layer (edge)

        #OBJECT ATTRIBUTES LATER COMPUTED
        #self.depth_hidden_layers  = the depth (just the number of hidden layers NOT incl output layer)
        #self.learned_params  #what we learn using gradient descent in fit function IF WE DON'T DO STOCHASTIC
        #self.N  = the number of training instances fit to
        
        
            
    #initialize the weights in the constructor according to intialization type (parameter to tune)
    def initialize_parameters(self, init_type):
        #initialization of weights - hard coded to randomize or set to 0 but this is HP to tune
        if init_type == "RANDOM":
            #note extra col for BIAS 
            V = np.random.randn(self.D+1,self.M) * .01
            W = np.random.randn(self.M+1, self.C) * .01 #out of func intiialize weights and store initial versions

            #I think biases are actually just best incorm in X and W (extra dim)
            #bw = np.random.randn(self.M) * .01
            #TODO biases of last layer will be related to C?
            #blast = np.random.randn(self.C) * .01

        #elif init_type == "ZERO":
        #put other paramater intiialization here
        return V, W  #, bw, blast    

    #create layer list here for model
    #called in class intializer
    #if there are not hidden layers I thiiink this should work, TODO test for no HL case (Task 3.1) I think just log regr
    #returns a list of all layers
    def create_layers(self, hidden_activation_func_list, output_activation_func):
        #list of all layers
        layers_list = []

        #create first layer and add to list
        layers_list.append(l.InputEdge(self.V))

        #create hidden layers: length of passed activation funcs determines numbre of hidden layers
        #if it's the last index before the output func, weight matrix diff dimensions
        last_index = len(hidden_activation_func_list)-1
        for index, activation_function in enumerate(hidden_activation_func_list):
            weights = self.V if index != last_index else self.W
            hid_layer = l.HiddenLayer(activation_function)
            layers_list.append(hid_layer)
            hid_edge = l.Edge(weights)
            layers_list.append(hid_edge)

        #create output layer
        layers_list.append(l.OutputLayer(output_activation_func))
        return layers_list

    #Compute forward pass
    #parameters stored in layer object as needed
    #here I passed the parameterse to the forward_pass func directly rather than the model object
    #   incase we want to play around w stochastic vs full batch gradient descent
    def forward_pass(self, X):
        #scope in python is a function!
        for i,layer in enumerate(self.layers):
            input = X if i == 0 else z # X will be first layer input, otherwise it's the output, z, of last layer
            self.activations.append(input) #possible bug appending X but I don't think it'll matter
            z = layer.get_output(input)
        yh = z # last value computed is yh, rename for consistency
        return yh

    #def fit(self, X, Y, optimizer): #optimizer is GD in our case
    def fit(self, X):
        N,D = X.shape 
        self.N = N
        self.D = D

        #must add bias to X input, so now D + 1 width - an extra col?
        bias = np.ones((N,1), dtype=float)
        X = np.append(X, bias, axis=1)
 
        self.forward_pass(X)     

        #params0 = [v,w]
        #optimizer here is gd, passed to the fit function
        #self.learned_params = optimizer.run(gradient, x, y, params0) #pass grad , x, ,y, initial params 
        
        #returns optimized parameters
        return self
    
    #We've just built a function!
    #TODO ADD learned biases too
    #   shouldn't actually be necc to save v, w sepately if using stochastic GD
    #   but I'll keep it for now if we want to play around
    def predict(self, X): 
        Vlearned, Wlearned = self.learned_params
        yh = self.forward_pass(X , Vlearned, Wlearned)
        return yh #should be dim N