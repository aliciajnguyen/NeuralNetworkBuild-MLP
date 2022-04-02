import numpy as np
import matplotlib.pyplot as plt
import MLPModel as model
import Utilities as u
import ActivationFunctions as af

def experiment4():
    #data
    x = np.array([[1, 2], [3, 4], [2, 1], [5, 6]]) #each row is an input, so each inner bracket []
    # one hot encode : see https://edstem.org/us/courses/18448/discussion/1334861
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]) #one hot encoding
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

    yh_probs = model1.predict_probs(x)
    yh = model1.predict(x)

    print(y)
    print(yh_probs)
    print(yh)

    u.evaluate_acc(y, yh)


experiment4()

#no hidden layers test
def experiment5():
    #data
    x = np.array([[1, 2], [3, 4], [2, 1], [5, 6]]) #each row is an input, so each inner bracket []
    # one hot encode : see https://edstem.org/us/courses/18448/discussion/1334861
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]) #one hot encoding
    # D = 2
    # N = 4
    # M = 2
    # C = 3


#building the models for task 3-1
def Task3_1(X, Y):
    C = Y.shape[1] #TODO how many classes in dataset?
    N,D = X.shape 

    #1)no hidden layers
    hidlayer_activfunc_list1 = []
    output_activation1 = af.SoftMax()
    #create model object
    model3_1_1 = model.MLP(M=0, D=D, C=C, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1)

    #2)1 hidden layer, 128 hidden units
    hidlayer_activfunc_list2 = []
    hidlayer_activfunc_list2.append(af.ReLU())
    output_activation2 = af.SoftMax()
    #create model object
    model3_1_2 = model.MLP(M=128, D=D, C=C, hidden_activation_func_list=hidlayer_activfunc_list2, output_activation_func=output_activation2)

    #3)1 hidden layer, 128 hidden units
    hidlayer_activfunc_list3 = []
    hidlayer_activfunc_list3.append(af.ReLU())
    hidlayer_activfunc_list3.append(af.ReLU())
    output_activation3 = af.SoftMax()
    #create model object
    model3_1_3 = model.MLP(M=128, D=D, C=C, hidden_activation_func_list=hidlayer_activfunc_list3, output_activation_func=output_activation3)
