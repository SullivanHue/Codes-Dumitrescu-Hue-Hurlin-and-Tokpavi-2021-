
function output=adaptivePenalizedLogisticTree2SplitsALasso(explLearning,depLearning,...
                             explTest,depTest,depLearningQual,name_var_final)
[noo,KK]=size(explLearning);

%% trees with one split
IndicatorVarLearning = explLearning;
IndicatorVarTest = explTest;
DecisionRule = name_var_final;
for i=1:KK
  catvar=(ismember(i,depLearningQual)).*i;
  Inf_arbre=fitctree(explLearning(:,i),depLearning,...
       'MaxNumSplits',1,'MergeLeaves','off','categoricalpredictors',find(catvar~=0),...
       'Prune','off','MaxNumCategories',10);
  [~,~,node,~]  = Inf_arbre.predict(explLearning(:,i));
 [~,~,node1,~]  = Inf_arbre.predict(explTest(:,i));
  %
  U = zeros(noo,1);
  U(node==2)=1;
  %
  V = zeros(noo,1);
  V(node1==2)=1; 
  if ~ismember(0,sum(abs(IndicatorVarLearning-repmat(U,1,size(IndicatorVarLearning,2))))).*...
                  ~ismember(1,(sum(IndicatorVarLearning+repmat(U,1,size(IndicatorVarLearning,2)))==noo))
        IndicatorVarLearning = [IndicatorVarLearning U];  
       IndicatorVarTest = [IndicatorVarTest V]; 
        DecisionRule{end+1}= strcat(name_var_final(i),'<',num2str(Inf_arbre.CutPoint(1)));      
  end           
end
 
%% Trees with depth 2 for couples of predictors with indicator variables generations
for i=1:KK
     for j=i+1:KK
         catvar=(ismember([i,j],depLearningQual)).*[i,j];
         Inf_arbre=fitctree(explLearning(:,[i j]),depLearning,...
             'MaxNumSplits',2,'MergeLeaves','off','categoricalpredictors',find(catvar~=0),...
             'Prune','off','MaxNumCategories',10,'PredictorNames',name_var_final([i,j]));
         [~,~,node,~]  = Inf_arbre.predict(explLearning(:,[i j]));
        [~,~,node1,~]  = Inf_arbre.predict(explTest(:,[i j]));
         %
         %case 1
         if sum(ismember(find(~Inf_arbre.IsBranch),[3 4 5]))==3
           V1 = zeros(noo,1);
           V1(node==3)=1;
           V2 = zeros(noo,1);
           V2(node==4)=1;
           V1s = zeros(noo,1);
           V1s(node1==3)=1; 
           V2s = zeros(noo,1);
           V2s(node1==4)=1; 
           if ~ismember(0,sum(abs(IndicatorVarLearning-repmat(V1,1,size(IndicatorVarLearning,2))))).*...
                 ~ismember(1,(sum(IndicatorVarLearning+repmat(V1,1,size(IndicatorVarLearning,2)))==noo))
              IndicatorVarLearning = [IndicatorVarLearning V1]; 
             IndicatorVarTest = [IndicatorVarTest V1s]; 
              DecisionRule{end+1} = strcat(Inf_arbre.CutPredictor(1),'>=',num2str(Inf_arbre.CutPoint(1)));
           end
           if ~ismember(0,sum(abs(IndicatorVarLearning-repmat(V2,1,size(IndicatorVarLearning,2))))).*...
                  ~ismember(1,(sum(IndicatorVarLearning+repmat(V2,1,size(IndicatorVarLearning,2)))==noo))
                IndicatorVarLearning = [IndicatorVarLearning V2]; 
               IndicatorVarTest = [IndicatorVarTest V2s]; 
                DecisionRule{end+1} = strcat(strcat(Inf_arbre.CutPredictor(1),'<',num2str(Inf_arbre.CutPoint(1))),'\\',strcat(Inf_arbre.CutPredictor(2),'<',num2str(Inf_arbre.CutPoint(2))));
           end
         end
         %case 2
         if sum(ismember(find(~Inf_arbre.IsBranch),[2 4 5]))==3
           V1 = zeros(noo,1);
           V1(node==2)=1;
           V2 = zeros(noo,1);
           V2(node==4)=1;
           V1s = zeros(noo,1);
           V1s(node1==2)=1; 
           V2s = zeros(noo,1);
           V2s(node1==4)=1; 
           if ~ismember(0,sum(abs(IndicatorVarLearning-repmat(V1,1,size(IndicatorVarLearning,2))))).*...
                 ~ismember(1,(sum(IndicatorVarLearning+repmat(V1,1,size(IndicatorVarLearning,2)))==noo))
              IndicatorVarLearning = [IndicatorVarLearning V1]; 
             IndicatorVarTest = [IndicatorVarTest V1s]; 
              DecisionRule{end+1} = strcat(Inf_arbre.CutPredictor(1),'<',num2str(Inf_arbre.CutPoint(1)));
           end
           if ~ismember(0,sum(abs(IndicatorVarLearning-repmat(V2,1,size(IndicatorVarLearning,2))))).*...
                  ~ismember(1,(sum(IndicatorVarLearning+repmat(V2,1,size(IndicatorVarLearning,2)))==noo))
              IndicatorVarLearning = [IndicatorVarLearning V2]; 
             IndicatorVarTest = [IndicatorVarTest V2s]; 
              DecisionRule{end+1} = strcat(strcat(Inf_arbre.CutPredictor(1),'>=',num2str(Inf_arbre.CutPoint(1))),'\\',strcat(Inf_arbre.CutPredictor(3),'<',num2str(Inf_arbre.CutPoint(3))));
           end
         end
     end
end

%% run the adaptive Lasso with 10-fold cross validation
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

% %% marginal effects 
margEff=NaN(size(IndicatorVarLearning));
Coffssc=Coeffs(2:end,1);
for i=1:size(IndicatorVarLearning,1)
  for j=1:size(IndicatorVarLearning,2)
    margEff(i,j)=(Coffssc(j,1)*exp([1 IndicatorVarLearning(i,:)]*Coeffs))/...
           ((1+exp([1 IndicatorVarLearning(i,:)]*Coeffs))^2);
  end
end
   
% %Mean of marginal effects
memargEff=(1/size(IndicatorVarLearning,1))*sum(margEff);

%% output results

output.exogvarLearning=IndicatorVarLearning;
output.exogvarTest=IndicatorVarTest;
output.depvarLearning=depLearning;
output.depvarTest=depTest;
output.Coeffs=Coeffs;
output.predict_alassoLearning=predict_lassoLearning;
output.predict_alassoTest=predict_lassoTest;
output.OptCO_log_alasso=OptCO_log_lasso;
output.binaire_alasso=binaire_lasso;
output.activeVar=sum(Coeffs~=0);
output.activeVarPct=mean(Coeffs~=0);
output.marginalEffect=memargEff';
output.DecisionRule=DecisionRule;


