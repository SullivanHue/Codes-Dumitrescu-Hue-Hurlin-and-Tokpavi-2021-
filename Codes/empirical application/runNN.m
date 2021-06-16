
function [predictProbLearning,predictProbTest,predictClassTest]= ...
                   runNN(depLearning,explLearning,explTest)

%% Training SVM

restoredefaultpath
net = patternnet(10,'trainbfg');
net = train(net,explLearning',depLearning');
               
%% predictions
classifScore = (net(explLearning'))';
predictProbLearning=classifScore;                          
nbEvent = sum(depLearning==1);
oo = 0:0.001:1;
nbo = NaN(length(oo),1);
for i=1:length(oo)
  nbo(i)=sum(predictProbLearning>=oo(i));    
end
[~,idx]=min(abs(nbo-nbEvent));
OptCO_RF = oo(idx);
%
classifScore = (net(explTest'))';
predictProbTest=classifScore;
predictClassTest = (predictProbTest>=OptCO_RF);
predictClassTest=double(predictClassTest);

