#past fit

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
