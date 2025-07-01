clc; clear;
% Load data
document = {'glass_uni'};
load([char(document) '.mat']);
X = NormalizeFea(X,1);
Y(Y==0) = 2;

% Parameter settings
classnum = 6;
para.lambda1 = 1;
para.lambda2 = 1;
nSamples = size(X,1);
ind = crossvalind('Kfold', nSamples, 10);

% Initialization (store k values using a cell array)
pr_Y = zeros(nSamples,1);
Time = zeros(10,1);
Optimalk = cell(1,10);
SubOptimalk = cell(1,10);
for k = 1:10
    testindex = (ind == k);
    trainindex = ~testindex;
    [time, label, W] = CKLKNN(X(trainindex,:), Y(trainindex,:), X(testindex,:), para);
    [optimalkvalue,suboptimalkvalue] = Kentropy( W, X(trainindex,:),Y(trainindex,:), X(testindex,:), 10);
    pr_Y(testindex) = label;
    Time(k) = time;
    Optimalk{k} = optimalkvalue;
    SubOptimalk{k} = suboptimalkvalue;
end
% Evaluate results
Acc = Accuracy(pr_Y, Y);
sumTime = sum(Time);

fprintf('Accuracy: %8.5f\nRuntime: %8.5f\n', Acc, sumTime);