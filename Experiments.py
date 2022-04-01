import numpy as np
import matplotlib.pyplot as plt
import MLPModel as model
import Utilities as u
import ActivationFunctions as af

def experiment4():
    #data
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]) #each row is an input, so each inner bracket []
    # one hot encode : see https://edstem.org/us/courses/18448/discussion/1334861
    y = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]) #one hot encoding
    # D = 2
    # N = 4
    # M = 2
    # C = 3

    # 2 hidden unit layer = both Relu
    # output layer = softmax
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