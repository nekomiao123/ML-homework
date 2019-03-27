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

for c = 1:num_labels %num_labels Ϊ�߼��ع�ѵ�����ĸ�����num of logistic regression classifiersv
  all_theta(c,:) = fmincg(@(t)(lrCostFunction(t, X, (y == c),lambda)), initial_theta,options );
end

%����һ����5000��������ÿ��������400��������������ˣ�ģ�Ͳ����� ������401��Ԫ�ء�
%initial_theta = zeros(n + 1, 1); % ģ�Ͳ����ȵĳ�ʼֵ(n == 400)
%all_theta��һ��10*401�ľ���ÿһ�д洢��һ��������(ģ��)��ģ�Ͳ����� ������ִ������forѭ�����͵���fmincg�⺯�� ����� ����ģ�͵Ĳ����� �����ˡ�
%�����ÿ��ģ�͵Ĳ��������ȣ��Ϳ����� ѵ���õ�ģ����ʶ�������ˡ�����һ����������������(400�� feature variables) input instance��
%ÿ��ģ�͵ļ��躯��h��(i)(x) ���һ��ֵ(i = 1,2,...10)��ȡ��10��ֵ�����ֵ�Ǹ�ֵ����Ϊ���յ�ʶ������
%����g(h��(8)(x))==0.96 ���������е� g(h��(i)(x)) (i = 1,2,...10,�� i ������8) ������ʶ��Ľ��Ϊ ���� 8.

% =========================================================================


end
