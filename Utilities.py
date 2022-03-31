import numpy as np

#BACK PROP CALCULATIONS

#OUTPUT LAYER       
#def get_dy(self, Y, Yh):
#        dy = Yh - Y
#        return dy

#FINAL EDGE
#    def get_dw(self, z, dy, N):
#        dw = np.dot(z.T, dy)/N
#        return dw

# Hiden Layer
#def get_dz(dy, V):
#        dz = np.dot(dy, V.T)
#        return dz

#EDGE     #z will be x at first layer?
#    def get_dv(self, z, dz, N):
#        dv = np.dot(z.T, dz * z * (1 - z))/N #D x M  #TODO change this it's a combo , should bebelow?

#general deriv case
#def get_dv(self, z, dz, dzq, N):
#        dv = np.dot(z.T, dz * dzq)/N #D x M  #this is for log deriv in square brackets #TODO check vectorization
#        return dv 




#COST FUNCTIONS

#TODO add logsumexp?

def cat_cross_entropy(Y, Yh):
    return - np.multiply(Yh, np.log(Y)).sum() / Y.shape[0]

#build to avoid underflow ??:
#def cat_cross_entropy(Y, Yh):
#    cost = -np.mean(np.sum(u * y, 1) - logsumexp(u)

#see backprop slide 19 - eg classifier
#def cat_cross_entropy(x, y , w , v):
#    q  = np.dot(x,v)
#    z = logistic (q)
#    u = np.dot(z,w)
#    yh = softmax(u)
#    return -np.mean(np.sum(u * y, 1) - logsumexp(u))






#HIDDEN LAYERS

#activation function (for hidden layers)
#TODO make sure this is getting applied elementwise!
def reLu(x):
    #return max(0, x)
    return np.maximum(0, x)

#TODO tanh and leaky reLu

#FINAL LAYER

#activation function (for final layer) regression 
def identity(x):
    return x

#activation function (for final layer) multiclass classification
def softmax(x):
    #caution doesn't apply on correct axis
    #e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum()
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

#activation function (for final layer) binary classification
def logistic(x):
    return 1./ (1. + np.exp(-x))


#pre-vectorized version of cat_cross_entropy
# - (T * np.log(Y)).sum()

#kind of just rough versions of things being incorporated into the rest of the code:

#logistic = lambda z: 1./ (1 + np.exp(-z))
