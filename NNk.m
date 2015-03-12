function [theta,sizes] =  NNk(X,y,hidden_layer_size,MaxIter,lambda)
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

%% Some Useful Variables:
m = size(X,1); % number of training samples
n = size(X,2); % number of features (input_layer_size)
r = size(y,1); % number of classes (num_labels or output_layer_size)
h = hidden_layer_size; % number of hidden layer units (hidden_layer_size)
k= length(h);
%% Initializing Parameters
% In this part we implement a k-layer neural network. First we initialize 
% the weights of the neural network (randInitializeWeights.m needed)

sizes = [n,h,r];
theta_ind=zeros(1,k+2);

for i = 1:k+1
    theta_ind(i+1) =  theta_ind(i)+(sizes(i)+1)*sizes(i+1);
end
initial_nn_params =zeros(theta_ind(end),1);


for i = 1:k+1
    mytheta =randInitializeWeights(sizes(i), sizes(i+1));
    initial_nn_params((theta_ind(i)+1):theta_ind(i+1)) = mytheta(:);
    % Unroll parameters
end
%% Training NN
% The cost function for two layer NN is implemented in nnCostFunction.m
% To train the NN, we will now use "fmincg". These
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
%fprintf('\nTraining Neural Network... \n')

options = optimset('Display', 'off','MaxIter', MaxIter);

costFunction = @(p) nnkCostFunction(p, h,  X, y, lambda);

[theta, cost] = fmincg(costFunction, initial_nn_params, options);


end