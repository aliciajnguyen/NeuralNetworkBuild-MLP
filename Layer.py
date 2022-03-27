#object for each layer of our MLP


class Layer:

    def __init__(self, activation_function):
        #edges?
        self.activation_function = activation_function
