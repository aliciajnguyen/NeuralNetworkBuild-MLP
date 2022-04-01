import numpy as np
import matplotlib.pyplot as plt
import MLPModel as model
import Utilities as u
import ActivationFunctions as af

def experiment4():
    #data
    # D = 2 , N = 2
    x = np.array([[1, 2], [3, 4], [5, 6]]) #each row is an input, so each inner bracket []
    #y = np.array([0.3, 0.4, 0.3], [0.5, 0.3, 0.2], [0.2, 0.1, 0.7]) #TODO note currently Yh 3x3 - decide where to put argmax
    # or should it be?? see https://edstem.org/us/courses/18448/discussion/1334861
    y = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) #one hot encoding

    # N = #
    # M =3
    # C =3
    # 2 hidden unit layer = both Relu
    # output layer = softmax

    #define parameters for model
    #create ActivationFunction objects, put in a list, pass to MLP constructor
    hidlayer_activfunc_list3 = []
    hidlayer_activfunc_list3.append(af.ReLU())
    hidlayer_activfunc_list3.append(af.ReLU())

    output_activation = af.SoftMax()

    #create model object
    model1 = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list3, output_activation_func=output_activation)

    # fit model 
    model1.fit(x, y)

experiment4()