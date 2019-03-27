function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);%m is 5000
n = size(X, 2);%n is 400

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];%add a column 1

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

initial_theta = zeros(n + 1, 1);
 
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels %num_labels 为逻辑回归训练器的个数，num of logistic regression classifiersv
  all_theta(c,:) = fmincg(@(t)(lrCostFunction(t, X, (y == c),lambda)), initial_theta,options );
end

%我们一共有5000个样本，每个样本有400中特征变量，因此：模型参数θ 向量有401个元素。
%initial_theta = zeros(n + 1, 1); % 模型参数θ的初始值(n == 400)
%all_theta是一个10*401的矩阵，每一行存储着一个分类器(模型)的模型参数θ 向量，执行上面for循环，就调用fmincg库函数 求出了 所有模型的参数θ 向量了。
%求出了每个模型的参数向量θ，就可以用 训练好的模型来识别数字了。对于一个给定的数字输入(400个 feature variables) input instance，
%每个模型的假设函数hθ(i)(x) 输出一个值(i = 1,2,...10)。取这10个值中最大值那个值，作为最终的识别结果。
%比如g(hθ(8)(x))==0.96 比其它所有的 g(hθ(i)(x)) (i = 1,2,...10,但 i 不等于8) 都大，则识别的结果为 数字 8.

% =========================================================================


end
