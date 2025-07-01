function [time,Label,W] = CKLKNN(traindata,Ytrain,testdata,para)
%% one step Knn
%Ytrain represents the label of training data
X = traindata;   %Training data
Y = testdata;     %test data
lambda1 = para.lambda1;  %parameter alpha
lambda2 = para.lambda2;     %parameter beta
[n,d] = size(X);
[m,~] = size(Y);
e = ones(d,1);
YY = Y';
XX = X';
%% initialization
W = rand(n,m);
iter = 0;
obji = 1;
Wsum = sum(W,2);
m1=5;
kmMaxIter = 10;
kmNumRep = 1;
[~,anchor] = litekmeans(X',m1,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
clear kmMaxIter kmNumRep

[mm,~] = size(anchor);
k = 1;
for i =1:d
    for j =1:mm
        DD(i,j) = sum((XX(i,:) - anchor(j,:)).^2);
    end
end
for i =1:d
    for j =1:mm
        Z(i,j) = (DD(i,k+1) - DD(i,j))/(k*DD(i,k+1)-sum(DD(i,1:k)));
    end
end

deta = sum(Z);
Deta = diag(deta);
S = Z*pinv(Deta)*Z';
II= ones(d,d);
L = II - S;
while 1
    %% 求解b
    b = (1/d).*(e'*Y' - e'*X'*W);
    Ybb = Y' - (e*b);
    Yb = Ybb';
    %%  求出W
    %     dn = 0.5./(sqrt(sum(W.*W,2)+eps));
    %     N = diag(dn);
    
    for j =1:m
        DN{j} = 0.5./(sqrt(sum(W(:,j).*W(:,j),2)+eps));
        D{j} = diag(DN{j});
    end
    for jj =1:m
        W(:,jj) = pinv(X*X'+lambda1*D{jj}+lambda2*X* L*X')*(X*(YY(:,jj)- e*b(jj)));
    end
    WI = sqrt(sum(W.*W,2)+eps);
    W21 = sum(WI);
    
    %%
    iter = iter + 1;
    if  iter == 4,    break,     end
    
end
W(W<mean(W(:))) = 0;
tic;
for ii = 1:m
    idx = find(W(:,ii) ~= 0);
    ok(ii) = length(idx);
    Labels(ii) = knnclassify(Y(ii,:),X,Ytrain, ok(ii), 'euclidean', 'nearest');
end
Label = Labels';
time = toc;
end
