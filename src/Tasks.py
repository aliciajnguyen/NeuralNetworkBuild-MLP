#TASK3 HERE
#building the models for tasks in part 3
import numpy as np
import MLPModel as model
import Utilities as u
import ActivationFunctions as af
import DataLoad as dl

import numpy as np

#TASK 3_1: Varying numbers of depth  layers
#just pass actualy y, x values
#tons of hyper parameters to tune as input to fit function
#can also change weight initialization in model creation
def task3_1(Xtrain, Xtest, Ytrain, Ytest):
    print("++++++++++++++++++TASK 3_1: Varying number of depth  layers ++++++++++++++++++")
    Ctask = Ytrain.shape[1] #10 classes in FASHION-MINST dataset, should be one hot encoded
    Ntask,Dtask = Xtrain.shape 

    #############################################################################
    #1)no hidden layers
    hidlayer_activfunc_list1 = []
    output_activation1 = af.SoftMax()
    #create model object
    model3_1_1 = model.MLP(M=0, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1)
    # fit model 
    model3_1_1.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=550, dropout_p=None)
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
    model3_1_2 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list2, output_activation_func=output_activation2, parameter_init_type="ACTIVATION_SPECIFIC")
    # fit model 
    model3_1_2.fit(Xtrain, Ytrain, learn_rate=0.2, gd_iterations=500, dropout_p=None)
    Yh2 = model3_1_2.predict(Xtest)  
    #print stats
    model3_1_2.print_model_summary()
    u.evaluate_acc(Ytest, Yh2)
    #############################################################################

    #############################################################################
    #3)2 hidden layers, 128 hidden units
    hidlayer_activfunc_list3 = []
    hidlayer_activfunc_list3.append(af.ReLU())
    hidlayer_activfunc_list3.append(af.ReLU())
    output_activation3 = af.SoftMax()
    #create model object
    model3_1_3 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list3, output_activation_func=output_activation3, parameter_init_type="ACTIVATION_SPECIFIC")
    # fit model 
    model3_1_3.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=750, dropout_p=None)
    Yh3 = model3_1_3.predict(Xtest)  
    #print stats
    model3_1_3.print_model_summary()
    u.evaluate_acc(Ytest, Yh3)

#TASK 3_2: Different activations
def task3_2(Xtrain, Xtest, Ytrain, Ytest):
    print("++++++++++++++++++TASK 3_2: Different activation functions ++++++++++++++++++")

    Ctask = Ytrain.shape[1] #10 classes in FASHION-MINST dataset, should be one hot encoded
    Ntask,Dtask = Xtrain.shape 

    #############################################################################
    #1) 2 layer Tanh
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.tanh())
    hidlayer_activfunc_list1.append(af.tanh())
    output_activation1 = af.SoftMax()
    #create model object
    model3_2_1 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1, parameter_init_type="ACTIVATION_SPECIFIC")
    # fit model 
    model3_2_1.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=750, dropout_p=None)
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
    model3_2_2 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list2, output_activation_func=output_activation2, parameter_init_type="ACTIVATION_SPECIFIC")
    # fit model 
    model3_2_2.fit(Xtrain, Ytrain, learn_rate=0.1, gd_iterations=750, dropout_p=None)
    Yh2 = model3_2_2.predict(Xtest)  
    #print stats
    model3_2_2.print_model_summary()
    u.evaluate_acc(Ytest, Yh2)

#TASK 3_3: 2 Hidden Layers, Relu, with DROPOUT
#dropout_p = percentages is a list, length = # hidden layers
#probabilities represent proportion of neurons we KEEP
#TODO can change values for hyperparameter tuning
#for now default values for testing
def task3_3(Xtrain, Xtest, Ytrain, Ytest, layer_dropout_percents = [0.8, 0.8]):
    print("++++++++++++++++++TASK 3_3: DROPOUT: 2 Hidden Layers with Relu ++++++++++++++++++")

    Ctask = Ytrain.shape[1] #10 classes in FASHION-MINST dataset, should be one hot encoded
    Ntask,Dtask = Xtrain.shape 

    #############################################################################
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.ReLU())
    hidlayer_activfunc_list1.append(af.ReLU())
    output_activation1 = af.SoftMax()
    #create model object
    model3_2_1 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1, parameter_init_type="ACTIVATION_SPECIFIC")
    # fit model 
    model3_2_1.fit(Xtrain, Ytrain, learn_rate=0.2, gd_iterations=180, dropout_p=layer_dropout_percents)
    Yh1 = model3_2_1.predict(Xtest)  
    #print stats
    model3_2_1.print_model_summary()
    u.evaluate_acc(Ytest, Yh1)

    #############################################################################

#TASK 3_4: 2 Hidden Layers, Relu, with UNNORMALIZED IMAGES
#TODO must pass UNNORMALIZED IMAGES
def task3_4(Xtrain, Xtest, Ytrain, Ytest):
    print("++++++++++++++++++TASK 3_4: UNNORMALIZED DATA: 2 Hidden Layers with Relu ++++++++++++++++++")
    
    Ctask = Ytrain.shape[1] #10 classes in FASHION-MINST dataset, should be one hot encoded
    Ntask,Dtask = Xtrain.shape 

    #############################################################################
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.ReLU())
    hidlayer_activfunc_list1.append(af.ReLU())
    output_activation1 = af.SoftMax()
    #create model object
    model3_2_1 = model.MLP(M=128, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1)
    # fit model 
    model3_2_1.fit(Xtrain, Ytrain, learn_rate=0.3, gd_iterations=150, dropout_p=None)
    Yh1 = model3_2_1.predict(Xtest)  
    #print stats
    print("UNNORMALIZED DATA")
    model3_2_1.print_model_summary()
    u.evaluate_acc(Ytest, Yh1)
    #############################################################################

#TASK 3_6: Best We Can Do
#TODO Optimize
def task3_6(Xtrain, Xtest, Ytrain, Ytest, hid_units, epoques, learning_rate, dropout_list):
    print("++++++++++++++++++TASK 3_6: BEST: 2 Hidden Layers with Relu ++++++++++++++++++")
    Ctask = Ytrain.shape[1] #10 classes in FASHION-MINST dataset, should be one hot encoded
    Ntask,Dtask = Xtrain.shape 
    #############################################################################
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.ReLU())
    hidlayer_activfunc_list1.append(af.ReLU())
    hidlayer_activfunc_list1.append(af.ReLU())
    output_activation1 = af.SoftMax()
    #create model object
    model3_2_1 = model.MLP(M=hid_units, D=Dtask, C=Ctask, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation1, parameter_init_type = "ACTIVATION_SPECIFIC")
    # fit model 
    model3_2_1.fit(Xtrain, Ytrain, learning_rate, gd_iterations=epoques, dropout_p=dropout_list)
    Yh1 = model3_2_1.predict(Xtest)  
    #print stats
    model3_2_1.print_model_summary()
    u.evaluate_acc(Ytest, Yh1)
    print("Accuracy on TRAIN set (seen data):")
    Yh2 = model3_2_1.predict(Xtrain)  
    #print stats
    model3_2_1.print_model_summary()
    u.evaluate_acc(Ytrain, Yh2)
    #############################################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#SAMPLE DATA
def get_sample_data1():
    Xtrain = np.array([[1, 3, 9], [-1, -4, -2], [5, 7, 1], [0.1, 0.7, 0.4]]) #each row is an input, so each inner bracket []
    # one hot encode : see https://edstem.org/us/courses/18448/discussion/1334861
    Ytrain = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]) #one hot encoding
    Xtest = np.array([[9, 2, 4],  [-5, -7, -1], [0.2, 0.3, 0.7]])
    Ytest = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return Xtrain, Ytrain, Xtest, Ytest
# D = 2
# N = 4
# M = 2
# C = 3
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ACTUAL DATA
#Xtrain, Ytrain, Xtest, Ytest = get_sample_data1()
#Xtrain, Ytrain, Xtest, Ytest = dl.get_prepped_original_data_from_file()
Xtrain, Ytrain, Xtest, Ytest = dl.load_local() # load dataset

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#DATA LOAD COLAB
#Xtrain, Ytrain, Xtest, Ytest = dl.load_dataset() # load dataset
#Xtrain, Xtest = dl.prep_pixels(Xtrain, Xtest)

task3_1(Xtrain, Xtest, Ytrain, Ytest)
#task3_2(Xtrain, Xtest, Ytrain, Ytest)
#task3_3(Xtrain, Xtest, Ytrain, Ytest, layer_dropout_percents = [0.6, 0.0])
#task3_6(Xtrain, Xtest, Ytrain, Ytest, hid_units=200, epoques=1000, learning_rate=0.3, dropout_list=[0.5, 0.5, 0])
