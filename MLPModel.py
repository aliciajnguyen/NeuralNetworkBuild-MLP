import numpy as np
import GradientDescent as gd
import Utilities as u
import Layer as l

class MLP:
    
    #should take as input the activ func (eg Relu)
    #numbre of hidden layers (eg 2)
    #numbre of hidden units in hidden layers
    #initialize weights and biases
    #also def loss function here depending on selection of output layer
    #note if that hidden activation func list is empty, no hidden layers constructed
    def __init__(self, M, D, C, hidden_activation_func_list, output_activation_func, parameter_init_type = "RANDOM"):
        self.M = M     # M =number of hidden units in hidden layers (width)
        self.D = D     # D inputs (number of x inputs) CONFUSION WITH N
        self.C = C          #C outputs (number of classes)        


        #list of represent all layers in mlp
        self.layers = self.create_layers(hidden_activation_func_list, output_activation_func)
                 
        #TODO def loss function here based on output layer ? or pass to layer

        #W dim = C X M
        #V dim = M x D  - np weird ordering in matrix creation careful
        self.W, self.V = self.initialize_parameters(parameter_init_type)

        #OBJECT ATTRIBUTES LATER COMPUTED
        #self.depth_hidden_layers  = the depth (just the number of hidden layers NOT incl output layer)
        #self.learned_params  #what we learn using gradient descent in fit function IF WE DON'T DO STOCHASTIC

    #initialize the weights in the constructor according to intialization type (parameter to tune)
    def initialize_parameters(self, init_type):
        #initialization of weights - hard coded to randomize or set to 0 but this is HP to tune
        if init_type == "RANDOM":
            W = np.random.randn(self.M, self.C) * .01 #out of func intiialize weights and store initial versions
            V = np.random.randn(self.D,self.M) * .01
        elif init_type == "ZERO":
            W = np.zeros(self.M, self.C) 
            V = np.zeros(self.D,self.M)
        return W,V    

    #TODO def a function for telling constructor how many hidden layers / what they are
    #later can extend if we want diff output functions at different layers
    #return a list of hidden layer functions

    #create layer list here for model
    #called in class intializer
    #if there are not hidden layers I thiiink this should work, TODO test for no HL case (Task 3.1) I think just log regr
    def create_layers(self, hidden_activation_func_list, output_activation_func):

        #compute the depth (just the number of hidden layers NOT incl output layer)
        self.depth_hidden_layers=len(hidden_activation_func_list)

        #append the output activation func for ease
        hidden_activation_func_list.append(output_activation_func)

        #add each hidden layer
        #TODO might have to change this if we use decorator for layers
        hidden_layers_list = []
        for i, activation_function in enumerate(hidden_activation_func_list):
            isFirst = True if i == 0 else False
            weights = self.V if i == 0 else self.W
            isLast = True if i == len(hidden_activation_func_list)-1 else False
            layer = l.Layer(weights, activation_function, isFirstLayer=isFirst, isLastLayer=isLast)
            hidden_layers_list.append(layer)

        return hidden_layers_list

    #Compute forward pass
    #parameters stored in layer object as needed
    #here I passed the parameterse to the forward_pass func directly rather than the model object
    #   incase we want to play around w stochastic vs full batch gradient descent
    def forward_pass(self, X , V , W):

        #scope in python is a function!
        for i,layer in enumerate(self.layers):
            input = X if i == 0 else z # X will be first layer input, otherwise it's the output, z, of last layer
            z = layer.get_output(input)
            yh = z # last value computed is yh, rename for consistency
            return yh

    #TODO : confused about when we're running GD? 
    def fit(self, x, y, optimizer): #optimizer is GD in our case
        N,D = x.shape #TODO sus out the actual dimension of D here
        self.N = N

     
        #why tf is gradient defined here? : so this func has access to outter variables
        def gradient(x, y, params): #computes grad of loss wrt dparams ?
            v, w = params
            
            #TODO: ask in OH
            #I think we'll just forward pass once
            #and then backward pass until stop cond for stochastic GD

            #FORWARD
            ################################
            z = af.logistic(np.dot(x, v)) #N x M #forward pass
            yh = af.logistic(np.dot(z, w))#N #forward pass
            ################################
            
            #BACKWARD PASS
            ################################
            #compute the loss function (should be cat cross entropy )
            dy = yh - y #N #using L2 loss even tho doing class? shouldn't, right way is to use cross entropy loss
            #dy is L / L _ y but we use dy?  dy = dL/dy
            dw = np.dot(z.T, dy)/N #M #take grad step in backwards direction, dw= dy/dw
            dz = np.outer(dy, w) #N x M #compute 
            dv = np.dot(x.T, dz * z * (1 - z))/N #D x M 
            dparams = [dv, dw] #store gradients of two parametsrs in a list

            ################################
            return dparams
        
        #the intial weights + TODO biases ? cur initialized in run
        w = self.w
        v = self.v
        params0 = [v,w]

        #optimizer here is gd, passed to the fit function
        self.learned_params = optimizer.run(gradient, x, y, params0) #pass grad , x, ,y, initial params 
        
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