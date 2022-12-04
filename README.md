# NeuralNetworkBuild-MLP

This was a collaborative project to design a Multilayer Perceptron architecture (MLP), or a shallow feed-forward Neural Network, from scratch solely utilizing 
the numpy library. We sought a design that was capable of varying the width, number of hidden layers (depth) and type of activation functions. This model was deployed to optimally classify images from the Fashion-MNIST dataset. Experimenting with depth, width, optimization hyperparameters and weight initialization we ultimately achieved an accuracy of 84%. This was accomplished using a 3 hidden layer network with ReLU, Leaky ReLU and tanh activation functions. By varying the number of hidden layers 
we demonstrated that the capacity to model nonlinear functions, as provided by a hidden layer, was integral to our task. However, in contrasting these results with
experiments using a pre-built Convolutional Neural Network (CNN), we found that further enhancement of performance was only achieved by utilizing the 
structure-conscious capacities offered by the convolution layers of a CNN.

![image](https://user-images.githubusercontent.com/52705170/205467723-f32425b6-33c7-4562-95ac-018dc1c2bf6b.png)
