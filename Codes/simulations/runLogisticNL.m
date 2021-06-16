
function [beta,predictProbLearning,predictProbTest,predictClassTest,OptCO_log_reg,name_beta]= ...
               runLogisticNL(depLearning,explLearning,explTest,explLearning1,depLearningQual,name_var)
           
[TT,KK]=size(explLearning);

explLearningNL = explLearning;
explTestNL = explTest;

%% croisement de variables deux à deux
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

%% carré des variables quantitatives
nquant = size(explLearning1,2)-length(depLearningQual);
for i=1:nquant
    U = explLearning(:,i).^2;
    if ~ismember(0,sum(abs(explLearningNL-repmat(U,1,size(explLearningNL,2))))).*...
                  ~ismember(1,(sum(explLearningNL+repmat(U,1,size(explLearningNL,2)))==TT))    
      explLearningNL = [explLearningNL U];
      explTestNL = [explTestNL explTest(:,i).^2];    
    end
end

NN=ones(length(depLearning),1);
[log_reg,~,stats]=glmfit(explLearningNL,[depLearning NN],'binomial','link','logit');
beta=stats.beta;
name_beta=['Intercept' name_var]';
predictProbLearning=glmval(log_reg,explLearningNL,'logit');
predictProbTest=glmval(log_reg,explTestNL,'logit');
%
nbEvent = sum(depLearning==1);
oo = 0:0.001:1;
nbo = NaN(length(oo),1);
for i=1:length(oo)
  nbo(i)=sum(predictProbLearning>=oo(i));    
end
[~,idx]=min(abs(nbo-nbEvent));
OptCO_log_reg = oo(idx);
%
predictClassTest=(predictProbTest>=OptCO_log_reg);
predictClassTest=double(predictClassTest);


