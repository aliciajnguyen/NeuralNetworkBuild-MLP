import numpy as np
import matplotlib.pyplot as plt
import MLPModel as model
import Utilities as u
import ActivationFunctions as af

#FORWARD PASS
#try eg from this video: https://www.youtube.com/watch?v=YOlOLxrMUOw
#not reall testing vector dimensions
def experiment1():
    #data
    #x = np.array([[0.25, 0.02]])
    x = np.array([[0.25, 0.02]])
    y = np.array([[0, 1]])

    #just 1 instance here
    # M =2 
    # C =2
    # 1 hidden unit layer = logistic
    # 1 output layer = logistic

    #define parameters for model
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(u.logistic)

    output_activation = u.logistic

    #create model object
    #note have to figure out D value from X.shape
    #model1 = model.MLP(M, D, C, hidden_activation_func_list, output_activation_func, parameter_init_type = "RANDOM")
    model1 = model.MLP(M=2, D=2, C=2, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "EXP1")

    # fit model (no backprop now)
    model1.fit(x)

def experiment2():
    #data
    # D = 2 , N = 2
    x = np.array([[0.1, 0.2], [1, 2]])
    y = np.array([1, 0])

    # M =3
    # C =2
    # 1 hidden unit layer = logistic
    # 1 output layer = logistic

    #define parameters for model
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(u.logistic)

    output_activation = u.logistic

    #create model object
    #note have to figure out D value from X.shape
    #model1 = model.MLP(M, D, C, hidden_activation_func_list, output_activation_func, parameter_init_type = "RANDOM")
    model1 = model.MLP(M=2, D=2, C=2, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation)

    # fit model (no backprop now)
    model1.fit(x)

def experiment3():
    #data
    # D = 2 , N = 2
    x = np.array([[1, 2], [3, 4], [5, 6]]) #each row is an input, so each inner bracket []
    y = np.array([1, 0, 2])

    # N = #
    # M =3
    # C =3
    # 2 hidden unit layer = both Relu
    # output layer = softmax

    #define parameters for model
    hidlayer_activfunc_list3 = []
    hidlayer_activfunc_list3.append(u.reLu)
    hidlayer_activfunc_list3.append(u.reLu)

    output_activation = u.softmax

    #create model object
    #note have to figure out D value from X.shape
    #model1 = model.MLP(M, D, C, hidden_activation_func_list, output_activation_func, parameter_init_type = "RANDOM")
    model1 = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list3, output_activation_func=output_activation)

    # fit model (no backprop now)
    model1.fit(x)

#SHINY NEW FORM
def experiment4():
    #data
    # D = 2 , N = 2
    x = np.array([[1, 2], [3, 4], [5, 6]]) #each row is an input, so each inner bracket []
    y = np.array([1, 0, 2]) #TODO note currently Yh 3x3 - decide where to put argmax

    # N = #
    # M =3
    # C =3
    # 2 hidden unit layer = both Relu
    # output layer = softmax

    #define parameters for model
    hidlayer_activfunc_list3 = []
    hidlayer_activfunc_list3.append(af.ReLU())
    hidlayer_activfunc_list3.append(af.ReLU())

    output_activation = af.SoftMax()

    #create model object
    #note have to figure out D value from X.shape
    #model1 = model.MLP(M, D, C, hidden_activation_func_list, output_activation_func, parameter_init_type = "RANDOM")
    model1 = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list3, output_activation_func=output_activation)

    # fit model (no backprop now)
    model1.fit(x, y)

experiment4()