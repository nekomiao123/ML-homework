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

% labels��labels�е�ÿһ��Ԫ�ص�ֵ��ʾ�������ɵľ����е����������ڼ��У���
% �ڶ��������ĸ���Ӧ�ú�labels��һ���࣬ÿ��Ԫ�ص�ֵ��ʾ���Ƕ�ӦlabelsԪ�ص��е�λ��
% ������������ʾ��Ҫ���õ�ֵ
% ���磺���labels = [1,2,3,1,2,3]; ��ô�ڶ�������Ҳ��������Ԫ�أ�����Ϊ��[1,2,3,4,5,6]����
% ��������Ϊ[2,2,3,3,4,4], ��ô������3*6��С�ľ������ɵľ���Ϊ��
% ��2 0 0 3 0 0�� 
% ��0 2 0 0 4 0��
% ��0 0 3 0 0 4��
% labels��Ԫ�ص����ֵָ�������������ڶ���������Ԫ�ص����ֵָ������������
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M=theta*data;        % M����ָ�������ϵ�thetaת�ó���x���Ǹ�����
M = bsxfun(@minus, M, max(M, [], 1));  
%max(M,[],1)ȡM�и������Ԫ�أ����Ϊһ����������max(M,[],2)Ϊ�������Ԫ��  
M=exp(M);  
H = bsxfun(@rdivide, M, sum(M));  %��һ����ʽ3  
M=log(H);  
M=M.*groundTruth;  
cost=-1/numCases*sum(sum(M,1),2)+ lambda/2 * sum(sum(theta.^2));  %��ʽ1  
thetagrad=-1/numCases*(groundTruth-H)*data'+lambda * theta;  %��ʽ2  










% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

