function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

s = 0;
for i = 1 : m
    s = s + (-1 * y(i) * log(h_theta_function(X(i, :), theta)) ...
        - (1 - y(i)) * log(1 - h_theta_function(X(i, :), theta)));
end
% caculate the sum of regularized part
s_theta = 0;
for j = 2 : n
    s_theta = s_theta + (theta(j) * theta(j));
end

J = s / m + lambda * s_theta / (2 * m);

% ====caculate gradiend vector====
for i = 1 : n
    s = 0;
    for j = 1 : m
        s = s + (h_theta_function(X(j, :), theta) - y(i)) * X(j, i);
    end
    if i > 1
        s = s + lambda * theta(i);
    end
    grad(i) = s / m;
end
% =============================================================

end

% h_theta_function
% h(x) = g(X * theta);
% g(z) = 1 / (1 + exp(-z));
function h = h_theta_function(x, theta)
% x caculate h(x) = g(Xt * theta), g is sigmoid function

h = 0;
h = sigmoid(x * theta);
end
