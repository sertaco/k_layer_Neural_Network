
clear ; close all; clc
%% Training set for XOR operator


X = [0 0 ...
    ;0 1 ...
    ;1 0 ...
    ;1 1];
y = [1 0 0 1];

%% Training a two layer NN with 1 hidden layers
%  As expected xor can be imitated with atleast one hidden layer.

%  With only 1 hidden layer and 400 iterations sometimes I don't get XOR 
% (depending on random initial theta's). Increasing the number of hidden layers 
%guarantee the fit. (With h=4 it's almost always trained well)


hidden_layer_size = [4 4];
lambda = 0;
MaxIter = 100;
cumacc=0;

[theta, sizes] =  NNk(X,y,hidden_layer_size,MaxIter,lambda);


%% Checking the results for the training set
pred = predictNNk(theta,sizes, X);

acc= mean(double(pred == y)) * 100;



fprintf('\nTraining Set Accuracy: %f\n',acc);


