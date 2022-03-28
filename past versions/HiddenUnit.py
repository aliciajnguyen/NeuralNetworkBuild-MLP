#USE VECTORIZATION INSTEAD

#import numpy as np

#do we really want this? might be v slow
#class HiddenUnit:

#    def __init__(self, w, b, layer, activation_function):
#        self.layer = layer      #belongs to a layer
#        self.activation_function = activation_function

#        self.w = w #weights : could be w or v depending on if 
#        self.b = b # bias
        #self.output is output of this HU

        #some helpful derivative things here?

        #called z and not x bc first layer prob separate
#        def compute_output(self, x):
#            output = self.activation_function(np.dot(self.w, z))