    #perform the backward pass
    def backward_pass(self, X, Y, Yh):

        #parameter_gradients = [] 
        parameter_gradients = collections.deque() #we'll pass this to gradient descent later, param grads per layer

        #careful, we're popping off actual layers - might want to make a copy here if we need to repeat backprop
        layer_activations = self.activations

        #last layer: error calc differently
        last_layer = self.layers.pop()
        Y = layer_activations.pop()
        input_grad = last_layer.get_input_grad(Y, Yh)
        input = layer_activations[-1]
        grads = last_layer.get_params_grad(X, output_grad)
        parameter_gradients.appendleft(grads)
        output_grad = input_grad
        
        #now for the remaining layers propogate part deriv calc backwards
        for layer in reversed(self.layers):

            output = layer_activations.pop()  #activations of the last layer on the stack
            input_grad = layer.get_input_grad(output, output_grad)             #hidden layer calc
            input = layer_activations[-1] # Get the input of this layer (activations of the previous layer)
   
            # Compute the layer parameter gradients used to update the parameters
            grads = layer.get_params_grad(input, output_grad)
            parameter_gradients.appendleft(grads)
            
            #use input grad of this layer to comp gradient of previous layer
            output_grad = input_grad

        return list(parameter_gradients)  # Return the parameter gradients
