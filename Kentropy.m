function [ optimalkvalue, suboptimalkvalue] = Kentropy( W, Xtrain,Ytrain, Xtest, number)

[n,~] = size(Xtrain);
[m,~] = size(Xtest);
suboptimalkvalue = zeros(number,m);
for ii = 1:m
     b = [];
    idx = find(W(:,ii) ~= 0);
    optimalkvalue(ii) = length(idx);
    majoritylabel = knnclassify(Xtest(ii,:),Xtrain,Ytrain, optimalkvalue(ii), 'euclidean', 'nearest');
    for jj =  optimalkvalue(ii)+1 : n
        majoritylabel1 = knnclassify(Xtest(ii,:),Xtrain,Ytrain, jj, 'euclidean', 'nearest');
        if majoritylabel1 == majoritylabel
            a = jj;
            b = vertcat(b,a);
            a = [];
        end
        if length(b) == number
            suboptimalkvalue(:,ii) = b;
        break;     
        end
    end
end

end

