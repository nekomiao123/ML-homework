function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

% labels：labels中的每一个元素的值表示在所生成的矩阵中的行数（即第几行），
% 第二个参数的个数应该和labels的一样多，每个元素的值表示的是对应labels元素的列的位置
% 第三个参数表示所要放置的值
% 例如：如果labels = [1,2,3,1,2,3]; 那么第二个参数也得有六个元素，比如为：[1,2,3,4,5,6]，第
% 三个参数为[2,2,3,3,4,4], 那么会生成3*6大小的矩阵，生成的矩阵为：
% 【2 0 0 3 0 0】 
% 【0 2 0 0 4 0】
% 【0 0 3 0 0 4】
% labels里元素的最大值指定最大的行数，第二个参数里元素的最大值指定最大的列数。
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M=theta*data;        % M就是指数函数上的theta转置乘以x的那个部分
M = bsxfun(@minus, M, max(M, [], 1));  
%max(M,[],1)取M中各列最大元素，结果为一个行向量；max(M,[],2)为各行最大元素  
M=exp(M);  
H = bsxfun(@rdivide, M, sum(M));  %归一化公式3  
M=log(H);  
M=M.*groundTruth;  
cost=-1/numCases*sum(sum(M,1),2)+ lambda/2 * sum(sum(theta.^2));  %公式1  
thetagrad=-1/numCases*(groundTruth-H)*data'+lambda * theta;  %公式2  










% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

