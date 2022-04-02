#TASK3 HERE
#building the models for task 3-1
from tkinter import Y
import numpy as np
import MLPModel as model
import Utilities as u
import ActivationFunctions as af

num_classes_fm = 10 #classes in the fashion minst dataset

#TASK 3_1: Varying numbers of depth  layers
#just pass actualy y, x values
#tons of hyper parameters to tune as input to fit function
#can also change weight initialization in model creation
#NOTE if you want to test with to sets, change M value
def task3_1(Xtrain, Xtest, Ytrain, Ytest):
    Ctask = Ytrain.shape[1] #10 classes in FASHION-MINST dataset, should be one hot encoded
    Ntask,Dtask = Xtrain.shape 

    #############################################################################
    #1)no hidden layers
    hidlayer_activfunc_list1 = []
    output_activation1 = af.SoftMax()
    #create model object
    model3_1_1 = model.MLP(M=0, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1)
    # fit model 
    model3_1_1.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    Yh1 = model3_1_1.predict(Xtest)  
    #print stats
    model3_1_1.print_model_summary()
    u.evaluate_acc(Ytest, Yh1)
    #############################################################################

    #############################################################################
    #2)1 hidden layer, 128 hidden units
    hidlayer_activfunc_list2 = []
    hidlayer_activfunc_list2.append(af.ReLU())
    output_activation2 = af.SoftMax()
    #create model object
    model3_1_2 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list2, output_activation_func=output_activation2)
    # fit model 
    model3_1_2.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    Yh2 = model3_1_2.predict(Xtest)  
    #print stats
    model3_1_2.print_model_summary()
    u.evaluate_acc(Ytest, Yh2)
    #############################################################################

    #############################################################################
    #3)1 hidden layer, 128 hidden units
    hidlayer_activfunc_list3 = []
    hidlayer_activfunc_list3.append(af.ReLU())
    hidlayer_activfunc_list3.append(af.ReLU())
    output_activation3 = af.SoftMax()
    #create model object
    model3_1_3 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list3, output_activation_func=output_activation3)
    # fit model 
    model3_1_3.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    Yh3 = model3_1_3.predict(Xtest)  
    #print stats
    model3_1_3.print_model_summary()
    u.evaluate_acc(Ytest, Yh3)

#TASK 3_2: Different activations
def task3_2(Xtrain, Xtest, Ytrain, Ytest):
    Ctask = Ytrain.shape[1] #10 classes in FASHION-MINST dataset, should be one hot encoded
    Ntask,Dtask = Xtrain.shape 

    #############################################################################
    #1) 2 layer Tanh
    hidlayer_activfunc_list1 = []
    output_activation1 = af.tanh()
    output_activation1 = af.tanh()
    #create model object
    model3_2_1 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1)
    # fit model 
    model3_2_1.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    Yh1 = model3_2_1.predict(Xtest)  
    #print stats
    model3_2_1.print_model_summary()
    u.evaluate_acc(Ytest, Yh1)
    #############################################################################

    #############################################################################
    #2)2 layer leaky relu
    hidlayer_activfunc_list2 = []
    hidlayer_activfunc_list2.append(af.LeakyReLU())
    hidlayer_activfunc_list2.append(af.LeakyReLU())
    output_activation2 = af.SoftMax()
    #create model object
    model3_2_2 = model.MLP(M=2, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list2, output_activation_func=output_activation2)
    # fit model 
    model3_2_2.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    Yh2 = model3_2_2.predict(Xtest)  
    #print stats
    model3_2_2.print_model_summary()
    u.evaluate_acc(Ytest, Yh2)
    #############################################################################

#SAMPLE DATA
Xtrain = np.array([[1, 3, 9], [-1, -4, -2], [5, 7, 1], [0.1, 0.7, 0.4]]) #each row is an input, so each inner bracket []
# one hot encode : see https://edstem.org/us/courses/18448/discussion/1334861
Ytrain = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]) #one hot encoding

Xtest = np.array([[9, 2, 4],  [-5, -7, -1], [0.2, 0.3, 0.7]])
Ytest = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# D = 2
# N = 4
# M = 2
# C = 3

task3_1(Xtrain, Xtest, Ytrain, Ytest)
task3_2(Xtrain, Xtest, Ytrain, Ytest)
