import collections
import numpy as np
import GradientDescent as gd
import Utilities as u
import Layers as l
import GradientDescent as gd
import ActivationFunctions as af

np.random.seed(1)

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
        # computed for every activation function layer NOT linear transformation layer (edge)
        #instead get from layer's themselves to avoid indexing issues

        #OBJECT ATTRIBUTES LATER COMPUTED
        #num hidden_layers just len of activation_func_list
        #self.learned_params  #what we learn using gradient descent in fit function IF WE DON'T DO STOCHASTIC
        #self.N  = the number of training instances fit to

    #create layer list here for model
    #called in class intializer
    #if there are not hidden layers I thiiink this should work
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
                in_edge = edge

        layers_list.append(l.OutputLayer(output_activation_func, cost_function))         #create output layer
        return layers_list

    #Compute forward pass, outputs stored in layer object as needed
    def forward_pass(self, X):
        print("Input to MLP X")
        print(X)
        last_index = len(self.layers)-1
        for i,layer in enumerate(self.layers):
            input = X if i == 0 else z # X will be first layer input, otherwise it's the output, z, of last layer            
            z = layer.get_output(input) #TODO - does the final layer get appended?
            if isinstance(layer, l.HiddenLayer): self.activations.append(z) #only append these activations z for backprop calc
        yh = z # last value computed is yh, rename for consistency
        return yh
    
    #perform the backward pass
    #return list(parameter_gradients)  # Return the parameter gradients
    #calls a number of functions in utilities to do partial derivative calculations
    def backward_pass(self, X, Y, Yh):
        # Yh = N X C

        layers = self.layers                 #edges and activation layers
        activations = []
        activations.append(X)                #add x as the beginning inpu
        activations.extend(self.activations) #output of each layer
        
        params = collections.deque()         #a list of parameters for gradient decent later

        #last layer and edge special case:
        #compute dy = pderiv L wrt W
        final_layer = layers.pop(-1)            #get output layer at end of list
        dy = u.get_dy(Y, Yh)                    #pderiv(Loss) wrt u actually
        z = activations.pop()                   #need last activations - ENSURE that
        final_edge = layers.pop()               #get last edge with W
        dw = u.get_dw(z, dy, self.N)  

        params.append(dw)               #save for grad desc

        #for iterating in the following loop, will change ever loop
        err_from_above = dy             #cost so far, backprop
        params_from_above = dw          #relavent params, backprop, kinda

        #reverse the layers (propograte from back): encounter hidden unit layer, then edge, then next hidden unit layer, etc
        for layer in reversed(layers): 
            #to calculate the derivative of a hidden layer we'll need 4 terms:
            #    #pder L wrt y * pder y wrt u = error from above = err_from_above
            #    #pder u wrt z = weights from the output of this layer = params_from_above
            #    #pder z wrt q = the deriv of activ func with z (this layer's output) passed
                #pder q wrt V
            
            if isinstance(layer, l.HiddenLayer):
                #get err_from_above * pderiv u/pderiv z
                z = activations.pop(-1)  
                dz = u.get_dz(err_from_above, params_from_above)    #params_abv will be set in the last iteration   

                #get the deriv of this function, WILL CHANGE depending on activation func, so deriv of af held in layer for easy comp
                dzq = layer.get_af_deriv(z)

                #what we'll pass to next layer
                dq = np.dot(dz, dzq) # want only the 4 partial deriv terms necc for error for above to backpropogate
                err_from_above = dq

            else: # will be an edge
                #z will still hold the activations from hidden unit layer output 
                #dq will still hold derivative we want from last iteration
                #dz still holds as well
                dv = u.get_dv(z, dz, dzq, self.N) #z should be activations from last layer

                params.appendleft(dv)
                params_from_above = dv

        params = list(params)         #params was a deque for efficiency, change back to list
        return params

    def fit(self, X, Y, learn_rate=1, gd_iterations=100, dropout_p=0):
        N,D = X.shape 
        self.N = N
 
        #bias implementation: ADD COLS of 1 to x, (must stay 1 to relect weight value)
        bias = np.ones((N,1), dtype=float)
        X = np.append(X, bias, axis=1)

        #DEBUG
        #Yh = self.forward_pass(X)     
        #learned_params = self.backward_pass(X, Y, Yh)
        
        def gradient(X, Y, params):    
            Yh = self.forward_pass(X)     
            learned_params = self.backward_pass(X, Y, Yh)
            return learned_params
        
        #create GradientDescent obj here and pass it our HP's
        #take out of the method pass tf
        optimizer = gd.GradientDescent(learning_rate=learn_rate, max_iters=gd_iterations)

        #actually run GD
        #optimizer here is gd, passed to the fit function
        #self.learned_params = optimizer.run(gradient, x, y, params0) #pass grad , x, ,y, initial params 
        
        #decide how to store parameters for fit
        return self
    
    def predict(self, X): 
        Vlearned, Wlearned = self.learned_params
        yh = self.forward_pass(X , Vlearned, Wlearned)
        #softmax returns probs so much use argmax?
        #Yh = np.argmax(Yh, axis = 1)
        return yh 