
function [AUCTest,PCCTest,BSTest,KSTest,PGITest,missClassifCost,EconomicIndicators] = ...
                 computeEvalCriteriaCS_Logistic(depTest,predictProbTest,predictClassTest)

%% compute AUC
[~,~,~,AUCTest] = perfcurve(depTest,predictProbTest,'1'); %AUC Test

%% compute PCC (Proportion of correct classification)
confusionTest=confusionmat(depTest,predictClassTest); %confusion matrix
PCCTest=sum(diag(confusionTest))/size(depTest,1);

%% compute Brier Score
BSTest=mean((depTest-predictProbTest).^2);

%% compute KS statistic
Liste1=predictProbTest(depTest==1,1);
Liste0=predictProbTest(depTest==0,1);
[~,~,KSTest]=kstest2(Liste0,Liste1);

%% compute Partial Gini Index
TP=[];
FP=[];
for x=0:0.0001:0.4
    class=double((predictProbTest>=x));
    TP=[TP;sum(depTest==1&class==1)/sum(depTest==1)]; 
    FP=[FP;sum(depTest==0&class==1)/sum(depTest==0)];
end

FP=sortrows(FP);
TP=sortrows(TP);

Int_trap=0;
for i=1:(length(FP)-1)
    Int_trap=Int_trap+((FP(i+1,1)-FP(i,1))*(TP(i+1,1)+TP(i,1)))/2;
end
PGITest=(2*Int_trap)-1; %Int corresponds to AUC with a and b equal to 0 et 1, respectively
%(PGI+1)/2 is equal to the AUC if the bounds are 0 and 1

%% economic cost
FPR = confusionTest(1,2)/(confusionTest(1,2)+confusionTest(1,1));
FNR = confusionTest(2,1)/(confusionTest(2,1)+confusionTest(2,2));
CFP=1;
CFN=2:50;
missClassifCost = CFP*FPR+CFN*FNR;

TPR = confusionTest(2,2)/(confusionTest(2,2)+confusionTest(2,1));
TNR = confusionTest(1,1)/(confusionTest(1,1)+confusionTest(1,2));
EconomicIndicators = [FPR; FNR; TPR; TNR];


