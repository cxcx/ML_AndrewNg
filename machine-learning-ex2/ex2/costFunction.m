function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
k = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
s = 0;
for i = 1 : m
    s  = s + (-1 * y(i) * log(h_theta_function(X(i, :), theta)) \
    - (1 - y(i)) * log(1 - h_theta_function(X(i, :), theta)));
J = s / m;
end

for j = 1 : k
    s2 = 0;
    for i = 1 : m
        s2 = s2 + ((h_theta_function(X(i, :), theta) - y(i)) * X(i, j));
    end
    grad(j) = s2 / m;
end

end
% h_theta_function
% h(x) = g(X * theta);
% g(z) = 1 / (1 + exp(-z));
function h = h_theta_function(x, theta)
% x caculate h(x) = g(Xt * theta), g is sigmoid function

h = 0;
h = sigmoid(x * theta);
end
