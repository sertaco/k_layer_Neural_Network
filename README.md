# k_layer_Neural_Network
This is a self contained k-layer neural network implementation with gradient descent and backpropagation, inspired by a previous 2 layer NN implementation: https://github.com/sertaco/Two-Layer-Neural-Network-NN2.git

% NNk implements a k-layer neural network with variable input, hidden and
% output layer sizes. NNk trains this NN for the given sample matrix
% X and output matrix y. NNk can be used for multiclass classification.
% The number of layers is fixed to three. NNk utilizes backpropagation with
% gradient descent.

%Inputs:
%*X: training set with nXm dimensions, m: no of training samples,n: no of features
%*y: training output with rXm dimensions, r: no of classes
%hidden_layer_size: [a b c ...] e.g., b being the layer size of the second
%hidden layer. The number of elements of the size array determines the
%number of hidden layers (k)
%*MaxIter: maximum iteration in gradient descent of each training
%*lambda: regularization parameter.

%Needed m files:
% -randInitializeWeights.m
% -nnkCostFunction.m
% -sigmoidGradient.m
% -fmincg.m
% -houtk.m
% -sigmoid.m
% -gettheta.m
% -predictNNk.m
