import collections
from multiprocessing import reduction
from winreg import REG_LINK
import numpy as np
import GradientDescent as gd
import Utilities as u
import Layers as l
import GradientDescent as gd
import ActivationFunctions as af
import collections
from multiprocessing import reduction
import numpy as np

np.random.seed(1)

class MLP():
    #currently the parameter initialization is hardcoded to be random, but could be changed later
    #parameter_init_type = "RANDOM" => all parameters initialzed randomly
    #parameter_init_type = "ACTIV_SPEC" => activation specific:  all parameters of an edge initialized based on
    #note to tune these hyper parameters we'll have to create different objects
    def __init__(self, M, D, C, hidden_activation_func_list, output_activation_func, cost_function = u.cat_cross_entropy, parameter_init_type = "RANDOM"):
        self.M = M     # M =number of hidden units in hidden layers (width)
        self.C = C     # C outputs (number of classes)        
        self.D = D     # D inputs (number of x inputs) 
        
        #for reporting on model
        self.parameter_init_type = parameter_init_type 
        self.num_hid_layers = len(hidden_activation_func_list)             
        self.activation_functions = hidden_activation_func_list
        
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
        init_params = []    #list of parameters for each edge layer #TODO refactor, just for GD

        #dimensions for parameter matrices with bias additions (one's col added to X too)
        Dplusbias = self.D +1       #V dim = (D+1, M)
        Mplusbias = self.M +1       #W dim = (M+1, C)

        #only for making variable width layers
        #M_last = Dplusbias          #for param initialization based on activation func, need width last layer

        #account for case with no hidden layers (log regression)
        if hidden_activation_func_list == None or len(hidden_activation_func_list) == 0  or self.M==0:
            spec_final_edge = l.Edge(Dplusbias, self.C)
            layers_list.append(spec_final_edge)       #create first edge (from X to first HU) and add to list
            init_params.append(spec_final_edge.get_params())
        else:
            #create hidden layers: length of passed activation funcs determines numbre of hidden layers

            #first edge has special dimensions
            edge = l.Edge(Dplusbias, Mplusbias) #TODO might need to make this M+1 for matrix mult
            layers_list.append(edge)

            final_index = len(hidden_activation_func_list)-1
            for i,activation_function in enumerate(hidden_activation_func_list):             

                hid_layer = l.HiddenLayer(activation_function) 
                layers_list.append(hid_layer)

                #BY DEFAULT "RANDOM" gives a sampling of gaussian dist (np.random.randn)
                #now if initialization type is activation specific, ensure weights initialized
                #based on hidden layer activation function
                #only gives best initializations for Relu, tanh, and leaky ReLu
                #note relies on standard width, otherwise change M to M_last
                if init_type == "ACTIVATION_SPECIFIC":
                    params = edge.get_params()
                    #params = params * 0.1
                    size_last_layer = Mplusbias if i != 0 else Dplusbias
                    params_custom = activation_function.param_init_by_activ_type(params, size_last_layer)
                    edge.set_params(params_custom)
                elif init_type == "AROUND_ZERO":
                    params = edge.get_params()
                    params_custom = params * 0.1
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
    
    def print_model_summary(self):
        print("-----Model summary:------------------")
        print(f'Number of Instances Trained On:  N = {self.N}')
        print(f'Number of Inputs Trained On:  D = {self.D}')
        print(f'Number of Hidden Units:  M = {self.M}')
        print(f'Number of Classes:  C = {self.C}')
        print(f'Parameter Initialization Type:  {self.parameter_init_type}')
        print(f'Gradient Descent Learning Rate: {self.learn_rate}')
        print(f'Gradient Descent Iterations: {self.gd_iterations}')
        print(f'Layer Dropout Keep Unit Percentages: {self.dropout_p}') #TODO dropout p will be per layer
        print(f'Number of Hidden Units Layers: {self.num_hid_layers}')
        print("Activation Functions: ")
        for af in self.activation_functions:
            print(type(af).__name__)


    #Compute forward pass
    def forward_pass(self, X):
        self.activations = []
        if self.dropout_p != None: keep_probs = self.dropout_p.copy()    #need to store for different descents
        #print("Input to MLP X") #debug forward pass
        #print(X)
        last_index = len(self.layers)-1
        for i,layer in enumerate(self.layers):
            input = X if i == 0 else z            # X will be first layer input, otherwise it's the output, z, of last layer            
            z = layer.get_output(input)
            #########
            if self.dropout_p != None and isinstance(layer, l.HiddenLayer):
              keep_prob_p  = keep_probs.pop(0)
              if keep_prob_p != 0:                                    #remove from list for this iteration
                drop_mask = (np.random.rand(*z.shape) < keep_prob_p) / keep_prob_p    #create dropout mask, invert to make predict scaling unnecc
                z *= drop_mask # drop!
            ##########
            if isinstance(layer, l.HiddenLayer): self.activations.append(z)       #only append these activations z for backprop calc
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

    #Hyperparameter: sdropout_p will be a list of dropout percentages for each layer
    def fit(self, X, Y, learn_rate=0.1, gd_iterations=50, dropout_p=None):
        self.N = X.shape[0]

        #for printing statistics
        self.learn_rate = learn_rate
        self.gd_iterations = gd_iterations
        self.dropout_p = dropout_p

        #bias implementation: ADD COLS of 1 to x, (must stay 1 to relect weight value)
        bias = np.ones((self.N,1), dtype=float)
        X = np.append(X, bias, axis=1)
        
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

        bias = np.ones((N,1), dtype=float)      #must add bias
        X = np.append(X, bias, axis=1)
        yh = self.forward_pass(X)               #compute through layers of functions

        return yh     

    def predict(self, X): 
        N = X.shape[0]

        bias = np.ones((N,1), dtype=float)      #must add bias
        X = np.append(X, bias, axis=1)
        yh_probs = self.forward_pass(X)         #compute through layers of functions

        def one_hot(row):
            #need to use argmax because it will break ties for us
            prediction_index = np.argmax(row, axis = 0) #get the index of most prob class, axis 0 bc single row
            row.fill(0) #in place set all values to 0
            row[prediction_index] =1

            return row
        
        yh =np.apply_along_axis(one_hot, 1, yh_probs)
        
        return yh 

    #### MINI BATCH GRADIENT DESCENT
"""
    def get_minibatch_grad(model, X_train, y_train):
      xs, hs, errs = [], [], []

      for x, cls_idx in zip(X_train, y_train):
          h, y_pred = forward(x, model)

          # Create probability distribution of true label
          y_true = np.zeros(n_class)
          y_true[int(cls_idx)] = 1.

          # Compute the gradient of output layer
          err = y_true - y_pred

          # Accumulate the informations of minibatch
          # x: input
          # h: hidden state
          # err: gradient of output layer
          xs.append(x)
          hs.append(h)
          errs.append(err)

      # Backprop using the informations we get from the current minibatch
      return backward(model, np.array(xs), np.array(hs), np.array(errs))
   
    def sgd_step(model, X_train, y_train):
      grad = get_minibatch_grad(model, X_train, y_train)
      model = model.copy()

      # Update every parameters in our networks (W1 and W2) using their gradients
      for layer in grad:
          # Learning rate: 1e-4
          model[layer] += 1e-4 * grad[layer]

      return model

    def sgd(model, X_train, y_train, minibatch_size):
      for iter in range(n_iter):
          print('Iteration {}'.format(iter))

          # Randomize data point
          X_train, y_train = shuffle(X_train, y_train)

          for i in range(0, X_train.shape[0], minibatch_size):
              # Get pair of (X, y) of the current minibatch/chunk
              X_train_mini = X_train[i:i + minibatch_size]
              y_train_mini = y_train[i:i + minibatch_size]

              model = sgd_step(model, X_train_mini, y_train_mini)

      return model"""