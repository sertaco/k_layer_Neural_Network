function [p,inp] = predictNNk(theta,sizes, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
% X must be a n X l matrix where n is the number of features, l is the
% number of samples to be predicted.

% Useful values

m = size(X, 1);
kk = size(sizes,2)-1;

inp = X';
for i = 1:kk
    %inp = sigmoid(gettheta(theta,sizes,i)*[ones(1,m);inp]);
    inp = sigmf(gettheta(theta,sizes,i)*[ones(1,m);inp],[1 0]);
end
    


p = inp>0.5;
end