import numpy as np
import MLPModel as model
import Utilities as u
import ActivationFunctions as af

def experiment1(x, y):
    # 2 hidden unit layer = both Relu
    # output layer = softmax
    #create ActivationFunction objects, put in a list, pass to MLP constructor
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.ReLU())
    hidlayer_activfunc_list1.append(af.ReLU())
    output_activation = af.SoftMax()

    #create model object
    a_model = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "RANDOM")

    # fit model 
    a_model.fit(x, y, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    yh = a_model.predict(x)  

    #print stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)

#different hyperparameters
def experiment2(x, y):
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.ReLU())
    hidlayer_activfunc_list1.append(af.ReLU())
    output_activation = af.SoftMax()
    #create model object
    a_model = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "ACTIVATION_SPECIFIC")
    # fit model 
    a_model.fit(x, y, learn_rate=0.6, gd_iterations=50, dropout_p=None)
    yh = a_model.predict(x)
    #get stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)

#tanh
def experiment3(x, y):
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.tanh())
    hidlayer_activfunc_list1.append(af.tanh())
    output_activation = af.SoftMax()
    #create model object
    a_model = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "ACTIVATION_SPECIFIC")
    # fit model 
    a_model.fit(x, y, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    yh = a_model.predict(x)
    #get stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)

#leaky relu rand init
def experiment4(x, y):
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    output_activation = af.SoftMax()
    #create model object
    a_model = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "RANDOM")
    # fit model 
    a_model.fit(x, y, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    yh = a_model.predict(x)
    #get stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)

#leaky relu activation specific init
def experiment5(x, y):
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    output_activation = af.SoftMax()
    #create model object
    a_model = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "ACTIVATION_SPECIFIC")
    # fit model 
    a_model.fit(x, y, learn_rate=0.1, gd_iterations=50, dropout_p=None)
    yh = a_model.predict(x)
    #get stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)

#leaky relu activation specific init, gd its = 200
def experiment6(x, y):
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    output_activation = af.SoftMax()
    #create model object
    a_model = model.MLP(M=2, D=2, C=3, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "ACTIVATION_SPECIFIC")
    # fit model 
    a_model.fit(x, y, learn_rate=0.1, gd_iterations=200, dropout_p=None)
    yh = a_model.predict(x)
    #get stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)

#try collapsing dimensions (setting M < D)
#dataset 2
def experiment7(x, y):
    hidlayer_activfunc_list1 = []
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    hidlayer_activfunc_list1.append(af.LeakyReLU())
    output_activation = af.SoftMax()

    #create model object
    a_model = model.MLP(M=2, D=3, C=3, hidden_activation_func_list=hidlayer_activfunc_list1, output_activation_func=output_activation, parameter_init_type = "ACTIVATION_SPECIFIC")

    # fit model 
    a_model.fit(x, y, learn_rate=0.1, gd_iterations=100, dropout_p=None)
    yh = a_model.predict(x)  

    #print stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)

#NO HIDDEN LAYER
#dataset 1
def experiment8(x, y):
    hidlayer_activfunc_list = [] #we'll just pass empty list
    output_activation = af.SoftMax()

    #create model object
    a_model = model.MLP(M=0, D=3, C=3, hidden_activation_func_list=hidlayer_activfunc_list, output_activation_func=output_activation, parameter_init_type = "ACTIVATION_SPECIFIC")

    # fit model 
    a_model.fit(x, y, learn_rate=0.1, gd_iterations=100, dropout_p=None)
    yh = a_model.predict(x)  

    #print stats
    a_model.print_model_summary()
    u.evaluate_acc(y, yh)


#data1
x = np.array([[1, 2], [3, 4], [2, 1], [5, 6]]) #each row is an input, so each inner bracket []
# one hot encode : see https://edstem.org/us/courses/18448/discussion/1334861
y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]) #one hot encoding
# D = 2
# N = 4
# M = 2
# C = 3

#data2
x2 = np.array([[1, 3, 9], [-1, -4, -2], [5, 7, 1], [0.1, 0.7, 0.4]]) #each row is an input, so each inner bracket []
# one hot encode : see https://edstem.org/us/courses/18448/discussion/1334861
y2 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]) #one hot encoding
# D = 2
# N = 4
# M = 2
# C = 3

#data 1
#experiment1(x, y)
#experiment2(x, y)
#experiment3(x, y)
#experiment4(x, y)
#experiment5(x, y)
#experiment6(x, y)
#experiment7(x2, y2) #data 2
experiment8(x2, y2)