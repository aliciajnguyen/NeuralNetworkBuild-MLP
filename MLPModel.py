import numpy as np
import ActivationFunctions as af
import GradientDescent as gd
import Utilities as u


class MLP:
    
    #should take as input the activ func (eg Relu)
    #numbre of hidden layers (eg 2)
    #numbre of hidden units in hidden layers
    #initialize weights and biases
    #just keep defaults for eg for now (we have logistic as all functions just this is how mlp tutorial done)
    #also def loss function here depending on selection of output layer
    def __init__(self, M = 64, parameter_init_type = "random", num_hidden_layers=1, hidden_activation_function = af.logistic, output_activation_function = af.logistic):
        self.M = M #number of hidden units in hidden layers (width)
        #self.D = D #num of "features" which will be pixel given out P3 dataset?  / number of inputs x_D ? 
        
        #layers
        self.num_hidden_layers = num_hidden_layers #does not include number of input or output layers
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

        #instead just represent as a list or array of Layer objects, that last will be out output
        

        self.hidden_layers= self.hidden_layers(num_hidden_layers, hidden_activation_function) #LIST of hidden activation functions: change later to incor mult inner activations, so we can adjust depth
        self.w, self.v = self.initialize_parameters(parameter_init_type)
        #self.learned_params  #what we learn using gradient descent in fit function

    #initialize the weights in the constructor according to intialization type (parameter to tune)
    def initialize_parameters(self, init_type):
        #initialization of weights - hard coded to randomize or set to 0 but this is HP to tune
        if init_type == "random":
            w = np.random.randn(self.M) * .01 #out of func intiialize weights and store initial versions
            v = np.random.randn(self.D,self.M) * .01
        elif init_type == "zero":
            w = np.zeros(self.M) 
            v = np.zeros(self.D,self.M)
        return w,v    

    #TODO def a function for telling constructor how many hidden layers / what they are
    #later can extend if we want diff output functions at different layers
    #return a list of hidden layer functions
    def hidden_layers(self, num_hidden_layers, hidden_activation_function):
        list_hidden_layers = []
        for i in range(num_hidden_layers):
            list_hidden_layers.append(hidden_activation_function)
        return list_hidden_layers

    #compute input through multiple functions, as multiplied with the weight parameters
    #will have to generalize this when I understand backprop more
    def forward_pass(self, x , v , w):
        #first hidden layer gets V as parameters
        z = hid_layer_func(np.dot(x, v))
        #compute rest of hidden layers
        for i, hid_layer_func in enumerate(self.hidden_layers):
            if i == 0: continue             #skip first hidden layer because special parameters (see above)
            z = hid_layer_func(np.dot(z, v))
        yh = self.output_activation_function(np.dot(z, w))
        return yh

    def fit(self, x, y, optimizer): #optimizer is GD in our case
        N,D = x.shape
        self.N = N
        self.D = D

        #why tf is gradient defined here? : so this func has access to outter variables
        def gradient(x, y, params): #computes grad of loss wrt dparams ?
            v, w = params
            
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
        
        #the intial weights + TODO biases
        w = self.w
        v = self.v

        #first back prop pass?

        params0 = [v,w]

        #optimizer here is gd, passed to the fit function
        self.learned_params = optimizer.run(gradient, x, y, params0) #pass grad , x, ,y, initial params 
        
        #returns optimized parameters
        return self
    

    def predict(self, x): #just a function
        v, w = self.learned_params
        #z = logistic(np.dot(x, v)) #N x M
        #yh = logistic(np.dot(z, w))#N
        yh = self.forward_pass(x , v , w)

        return yh