function [J, grad] = nnkCostFunction(theta, ...
                                   hidden_layer_size, ...
                                   X, y, lambda)
%nnkCostFunction Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = nnkCostFunction(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be an "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters 3 dimensional THETA matrix, the weight matrices
% for our 2 layer neural network

% Setup some useful variables
m = size(X,1); % number of training samples
n = size(X,2); % number of features (input_layer_size)
r = size(y,1); % number of classes (num_labels or output_layer_size)
h = hidden_layer_size; % number of hidden layer units (hidden_layer_size)
k=length(h);
sizes = [n,h,r];



% Forward propagation:         
AZ= houtk(X,theta,sizes);
y_=AZ{end};

J = 1/m*(-log(y_).*y-log(1-y_).*(ones(r,m)-y));
J=sum(sum(J));

% Regularization term:
for i=1:k+1
    mytheta = gettheta(theta,sizes,i);
    mytheta_ = mytheta(:,2:end);
    J = J + lambda/2/m*(sum(sum(mytheta_.^2)));
end


% Backpropagation


DEL =cell(k+1,1);
DEL{end} =  y_-y;

for i = 1:k    
    mytheta = gettheta(theta,sizes,k+2-i); 
    mytheta_ = mytheta(:,2:end);
    DEL{end-i} = (mytheta_'*DEL{end-i+1}).*sigmoidGradient(AZ{end-2*i-1});
    %[del2 del3 del4]
end

theta_grad = cell(1,k+1);

for i =1:k+1
    mytheta = gettheta(theta,sizes,i);
    mytheta_ = mytheta(:,2:end);
    theta_grad{i} = 1/m.*(DEL{i}*AZ{2*i-1}');
    % Regularization term
    theta_grad{i} =theta_grad{i} + lambda/m*[zeros(size(theta_grad{i},1),1),mytheta_]; 
end


% Unroll gradients


gradno=0;
for i = 1:k+1
    gradno =  gradno+size(theta_grad{i},1)*size(theta_grad{i},2);
end

grad=zeros(gradno,1);

flag=0;
for i=1:k+1
    a=theta_grad{i}(:);
    kamil=length(a);
    grad(flag+1:flag+kamil)=a;
    flag=flag+kamil;
end

end
