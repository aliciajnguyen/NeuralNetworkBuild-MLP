import numpy as np
import matplotlib.pyplot as plt
import MLPModel as model
import Utilities as u

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


#TODO
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

    #predict on model



experiment2()


    #1-D array
    #array_1 = np.array([1, 2, 3])
    #2-D array
    #array_2 = np.array(([1, 2, 3], [4, 5, 6], [7, 8, 9]))
