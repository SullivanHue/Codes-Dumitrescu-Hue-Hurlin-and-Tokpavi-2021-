
function [beta,predictProbLearning,predictProbTest,predictClassTest,OptCO_log_reg,name_beta]= ...
               runLogistic(depLearning,explLearning,explTest,name_var)

NN=ones(length(depLearning),1);
[log_reg,~,stats]=glmfit(explLearning,[depLearning NN],'binomial','link','logit');
beta=stats.beta;
name_beta=['Intercept' name_var]';
predictProbLearning=glmval(log_reg,explLearning,'logit');
predictProbTest=glmval(log_reg,explTest,'logit');
%
nbEvent = sum(depLearning==1);
oo = 0:0.001:1;
nbo = NaN(length(oo),1);
for i=1:length(oo)
  nbo(i)=sum(predictProbLearning>=oo(i));    
end
[~,idx]=min(abs(nbo-nbEvent));
OptCO_log_reg = oo(idx);
predictClassTest=(predictProbTest>=OptCO_log_reg);
predictClassTest=double(predictClassTest);


