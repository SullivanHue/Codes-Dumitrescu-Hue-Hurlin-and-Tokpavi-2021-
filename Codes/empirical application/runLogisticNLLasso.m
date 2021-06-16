
function output = runLogisticNLLasso(depLearning,explLearning,depTest,explTest,explLearning1,depLearningQual)
           
[TT,KK]=size(explLearning);

explLearningNL = explLearning;
explTestNL = explTest;

for i=1:KK
  for j=i+1:KK
     U = explLearning(:,i).*explLearning(:,j);
     if ~ismember(0,sum(abs(explLearningNL-repmat(U,1,size(explLearningNL,2))))).*...
                  ~ismember(1,(sum(explLearningNL+repmat(U,1,size(explLearningNL,2)))==TT))    
        explLearningNL = [explLearningNL U];
        explTestNL = [explTestNL explTest(:,i).*explTest(:,j)];
     end
  end
end

nquant = size(explLearning1,2)-length(depLearningQual);
for i=1:nquant
    U = explLearning(:,i).^2;
    if ~ismember(0,sum(abs(explLearningNL-repmat(U,1,size(explLearningNL,2))))).*...
                  ~ismember(1,(sum(explLearningNL+repmat(U,1,size(explLearningNL,2)))==TT))    
      explLearningNL = [explLearningNL U];
      explTestNL = [explTestNL explTest(:,i).^2];    
    end
end

%% run the adaptive Lasso with 10-fold cross validation 
IndicatorVarLearning = explLearningNL;
IndicatorVarTest = explTestNL;
cv_ridge = cv_penalized(glm_logistic(depLearning,IndicatorVarLearning),@p_ridge,'lambdamax',1,'folds',10);
beta0=cv_ridge.bestbeta;
cv =cv_penalized(glm_logistic(depLearning,IndicatorVarLearning),@p_adaptive,...
       'gamma',1,'adaptivewt',num2cell(beta0),'lambdamax',1,'folds',10);
Coeffs=cv.bestbeta(:,1);
predict_lassoLearning=glmval(Coeffs,IndicatorVarLearning,'logit');
predict_lassoTest=glmval(Coeffs,IndicatorVarTest,'logit');
%
nbEvent = sum(depLearning==1);
oo = 0:0.001:1;
nbo = NaN(length(oo),1);
for i=1:length(oo)
  nbo(i)=sum(predict_lassoLearning>=oo(i));    
end
[~,idx]=min(abs(nbo-nbEvent));
OptCO_log_lasso = oo(idx);
%
binaire_lasso=(predict_lassoTest>=OptCO_log_lasso);
binaire_lasso=double(binaire_lasso);

%% output results

output.exogvarLearning=IndicatorVarLearning;
output.exogvarTest=IndicatorVarTest;
output.depvarLearning=depLearning;
output.depvarTest=depTest;
output.Coeffs=Coeffs;
output.predict_alassoLearning=predict_lassoLearning;
output.predict_alassoTest=predict_lassoTest;
output.binaire_alasso=binaire_lasso;






