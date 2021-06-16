
function [predictProbLearning,predictProbTest,predictClassTest,depTestSVM]= ...
                   runSVM(depLearning,explLearning,explTest,depLearningQual,depTest)

%% Training SVM

%SVM predictions lead to NaN if a value of a categorical predictor of the test explicative variables is not in the values 
%of the learning explicative variables ==> NaN if it the case and remove the observation for the predictions
for j = 1:length(depLearningQual)
    freq_explTest = tabulate(explTest(:,depLearningQual(j)));
    freq_explLearning=arrayfun(@(z) sum(ismember(explLearning(:,depLearningQual(j)),z)),freq_explTest(1,1):freq_explTest(end,1))';
    freq_explLearning = [freq_explTest(:,1) freq_explLearning];
    for jj = 1 : size(freq_explTest,1)
        if freq_explTest(jj,2) > 0 && freq_explLearning(jj,2) == 0
            for jjj = 1:size(explTest,1)
                if explTest(jjj,depLearningQual(j)) == freq_explTest(jj,1)
                    explTest(jjj,depLearningQual(j)) = NaN;
                    depTest(jjj) = NaN;
                end
            end
        end  
    end
end

explTest = rmmissing(explTest);
depTestSVM = rmmissing(depTest);

SVMModel = fitcsvm(explLearning,depLearning,'CategoricalPredictors',depLearningQual,'Standardize',true,'KernelScale','auto');
SVMModel = fitPosterior(SVMModel,explLearning,depLearning);
               
%% predictions
[~,classifScore] = predict(SVMModel,explLearning);
predictProbLearning=classifScore(:,2);                          
nbEvent = sum(depLearning==1);
oo = 0:0.001:1;
nbo = NaN(length(oo),1);
for i=1:length(oo)
  nbo(i)=sum(predictProbLearning>=oo(i));    
end
[~,idx]=min(abs(nbo-nbEvent));
OptCO_RF = oo(idx);
%
[~,classifScore] = predict(SVMModel,explTest);
predictProbTest=classifScore(:,2);
predictClassTest = (predictProbTest>=OptCO_RF);
predictClassTest=double(predictClassTest);

