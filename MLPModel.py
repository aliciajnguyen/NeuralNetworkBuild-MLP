import collections
from multiprocessing import reduction
import numpy as np
import GradientDescent as gd
import Utilities as u
import Layers as l
import GradientDescent as gd
import ActivationFunctions as af

np.random.seed(1)

class MLP:
    
    #currently the parameter initialization is hardcoded to be random, but could be changed later
    #parameter_init_type = "RANDOM" => all parameters initialzed randomly
    #parameter_init_type = "ACTIV_SPEC" => activation specific:  all parameters of an edge initialized based on
    #note to tune these hyper parameters we'll have to create different objects
    def __init__(self, M, D, C, hidden_activation_func_list, output_activation_func, cost_function = u.cat_cross_entropy, parameter_init_type = "ACTIV_SPEC"):
        self.M = M     # M =number of hidden units in hidden layers (width)
        self.C = C     # C outputs (number of classes)        
        self.D = D     # D inputs (number of x inputs) CONFUSION WITH N
               
        #list of represent all layers in mlp
        self.layers = self.create_layers(hidden_activation_func_list, output_activation_func, cost_function, parameter_init_type)

        #OBJECT ATTRIBUTES LATER COMPUTED
        #self.init_params calculated in create_layers
        #num hidden_layers just len of activation_func_list
        #self.learned_params  #what we learn using gradient descent in fit function IF WE DON'T DO STOCHASTIC
        #self.N  = the number of training instances fit to 

        #computed for every activation function layer NOT linear transformation layer (edge)
        #created with each forward pass
        #self.activations = [] #a list of activations

    #create layer list here for model
    #called in class constructor
    #returns a list of all layers
    #hard coded so that all layers have to have the same number hidden units but this could be changed
    def create_layers(self, hidden_activation_func_list, output_activation_func, cost_function, init_type):
        layers_list = []    #list of all layers
        init_params = []    #list of parameters for each edge layer #TODO might not keep strategy

        #dimensions for parameter matrices with bias additions (one's col added to X too)
        Dplusbias = self.D +1       #V dim = (D+1, M)
        Mplusbias = self.M +1       #W dim = (M+1, C)

        #only for making variable width layers
        #M_last = Dplusbias          #for param initialization based on activation func, need width last layer

        #account for case with no hidden layers (log regression)
        if hidden_activation_func_list == None or len(hidden_activation_func_list) == 0  or self.M==0:
            spec_final_edge = l.Edge(self.D, self.C)
            layers_list.append(spec_final_edge)       #create first edge (from X to first HU) and add to list
            init_params.append(spec_final_edge.get_params())
            #no bias this case
        else:
            #create hidden layers: length of passed activation funcs determines numbre of hidden layers

            #first edge has special dimensions
            edge = l.Edge(Dplusbias, Mplusbias) #TODO might need to make this M+1 for matrix mult
            layers_list.append(edge)

            final_index = len(hidden_activation_func_list)-1
            for i,activation_function in enumerate(hidden_activation_func_list):             

                hid_layer = l.HiddenLayer(activation_function) 
                layers_list.append(hid_layer)

                #now if initialization type is activation specific, ensure weights initialized
                #based on hidden layer activation function
                #only gives best initializations for Relu, tanh, and leaky ReLu
                #note relies on standard width, otherwise change M to M_last
                if init_type == "ACTIV_SPEC":
                    params = edge.get_params()
                    params_custom = activation_function.param_init_by_activ_type(params, Mplusbias)
                    edge.set_params(params_custom)
                init_params.append(edge.get_params()) #append params only after they've been modified
                
                #create new edge
                #special case for the edges before the final output layer weight matrix W instead of V
                edge = l.Edge(Mplusbias, Mplusbias) if i != final_index else l.Edge(Mplusbias, self.C)
                layers_list.append(edge)

                #if using variable width update M_last here

        init_params.append(edge.get_params()) #append final edge that wasn't specially parameterized

        layers_list.append(l.OutputLayer(output_activation_func, cost_function))         #create output layer
        self.init_params = init_params #gave for GD later, not sure if we'll just use layers TODO

        return layers_list

    #Compute forward pass
    def forward_pass(self, X):
        self.activations = []
        print("Input to MLP X") #debug forward pass
        print(X)
        last_index = len(self.layers)-1
        for i,layer in enumerate(self.layers):
            input = X if i == 0 else z # X will be first layer input, otherwise it's the output, z, of last layer            
            z = layer.get_output(input)
            if isinstance(layer, l.HiddenLayer): self.activations.append(z) #only append these activations z for backprop calc
        yh = z # last value computed is yh, rename for consistency
        return yh
    
    #perform the backward pass
    #return list(parameter_gradients)  # Return the parameter gradients
    #note all dimensions here include bias ie M = M+1 from model creation
    def backward_pass(self, X, Y, Yh):
        #X = NxD
        #Y = N x C
        #W = M x C
        #V = D X M
        # Yh = N X C
        layers = self.layers.copy()             #edges and activation layers
        activations = self.activations.copy()   #outputs of each hidden layer
        activations.insert(0, X)                #add x as the beginning input, 
        
        params = collections.deque()            #a list of parameters for gradient decent later

        #last layer and edge special case:,     
        final_layer = layers.pop(-1)                #just to pop
        dy = Yh - Y                    #N x C       #pderiv(Loss) wrt y
        z = activations.pop()          #N x M       #need last activations 
        final_edge = layers.pop(-1)                 #get last edge with W
        params_from_above = final_edge.get_params() #get the weights for hidden layer calculations
        dw = np.dot(z.T, dy)/self.N    #M x C

        params.append(dw)                           #save for grad desc

        #for iterating in the following loop, will change every loop
        err_from_above = dy           #N x C  #cost so far, backprop
    
        #reverse the layers (propograte from back): encounter hidden unit layer, then edge, then next hidden unit layer, etc
        for layer in reversed(layers): 
            
            if isinstance(layer, l.HiddenLayer):
                #We need z^l-1 (output of last layer, not this layer)
                #so before we pop a new z, use the z-1 from the previous layer
                dzq = layer.get_af_deriv(z) #N x M #dzq should have dim of z

                z = activations.pop(-1)  #N x M
                dz = np.dot(err_from_above, params_from_above.T) #N x M    #params_abv will be set in the last iteration   
                err_from_above = dz #backprop error to next layer #TODO unclear about this, but I think correct

            else: # will be an edge
                #don't pop off activation because activations ONLY of HU layers
                #z will still hold the activations from hidden unit layer output 
                #dq will still hold derivative we want from last iteration
                #dz still holds as well
                dv =  np.dot(z.T, dz * dzq)/self.N #z should be activations from last layer DxM
                params_from_above = layer.get_params()  #layer will be edge, get V
                params.appendleft(dv)

        params = list(params)         #params was a deque for efficiency, change back to list
        return params

    def fit(self, X, Y, learn_rate=1, gd_iterations=50, dropout_p=0):
        N = X.shape[0]
        self.N =N

        #bias implementation: ADD COLS of 1 to x, (must stay 1 to relect weight value)
        bias = np.ones((N,1), dtype=float)
        X = np.append(X, bias, axis=1)

        #DEBUG
        #Yh = self.forward_pass(X)     
        #learned_params = self.backward_pass(X, Y, Yh)
        
        def gradient(X, Y, params):    
            Yh = self.forward_pass(X)     
            params = self.backward_pass(X, Y, Yh)
            return params
        
        #create GradientDescent obj here and pass it our HP's, then run GD
        optimizer = gd.GradientDescent(learning_rate=learn_rate, max_iters=gd_iterations)
        learned_params = optimizer.run(gradient, X, Y, self.init_params) #pass grad , x, ,y, initial params         
        
        #run through layers and set params
        for layer in reversed(self.layers):
            if isinstance(layer,l.Edge):
                layer.set_params(learned_params.pop())
        return self

    #returns the PROBABILITIES of classes 
    # (output of softmax rather than one hot encoding)
    def predict_probs(self, X): 
        N = X.shape[0]
        self.N =N

        bias = np.ones((N,1), dtype=float)      #must add bias
        X = np.append(X, bias, axis=1)
        yh = self.forward_pass(X)               #compute through layers of functions

        return yh     


    def predict(self, X): 
        N,D = X.shape
        self.N =N

        bias = np.ones((N,1), dtype=float)      #must add bias
        X = np.append(X, bias, axis=1)
        yh_probs = self.forward_pass(X)         #compute through layers of functions

        #yh = np.argmax(yh_probs, axis = 1, keepdims=True)      #softmax returns probs so much use argmax to get one hot encoding
        def one_hot(row):
            #need to use argmax because it will break ties for us
            prediction_index = np.argmax(row, axis = 0) #get the index of most prob class, axis 0 bc single row
            row.fill(0) #in place set all values to 0
            row[prediction_index] =1

             
            #prediction_max = np.max(row, axis = 0) #get the index of most prob class, axis 0 bc single row
            #f = np.vectorize(lambda e : 1 if e == prediction_index else 0) #make element wise, vector ready fun
            #discrete = lambda e : 1 if e == prediction_index else 0     #e each element
            # row = np.apply_along_axis(discrete, 0, row) # 0 axis bc single row
            return row
        
        yh =np.apply_along_axis(one_hot, 1, yh_probs)
        
        return yh 