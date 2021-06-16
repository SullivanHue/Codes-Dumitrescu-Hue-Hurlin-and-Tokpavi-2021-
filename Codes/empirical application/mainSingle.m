
function  output = mainSingle(N,whichdata) 

%% initialization
AUC_Logistic=NaN(N,2);
PCC_Logistic=NaN(N,2);
BS_Logistic=NaN(N,2);
KS_Logistic=NaN(N,2);
PGI_Logistic=NaN(N,2);
RelmissClassifCost_Logistic = NaN(N,2,49);
EconomicIndicators_Logistic = NaN(N,2,4);
%
AUC_LogisticNL=NaN(N,2);
PCC_LogisticNL=NaN(N,2);
BS_LogisticNL=NaN(N,2);
KS_LogisticNL=NaN(N,2);
PGI_LogisticNL=NaN(N,2);
RelmissClassifCost_LogisticNL = NaN(N,2,49);
EconomicIndicators_LogisticNL = NaN(N,2,4);
%
AUC_LogisticNL1=NaN(N,2);
PCC_LogisticNL1=NaN(N,2);
BS_LogisticNL1=NaN(N,2);
KS_LogisticNL1=NaN(N,2);
PGI_LogisticNL1=NaN(N,2);
RelmissClassifCost_LogisticNL1 = NaN(N,2,49);
EconomicIndicators_LogisticNL1 = NaN(N,2,4);
%
AUC_RF_v2=NaN(N,2);
PCC_RF_v2=NaN(N,2);
BS_RF_v2=NaN(N,2);
KS_RF_v2=NaN(N,2);
PGI_RF_v2=NaN(N,2);
RelmissClassifCost_RF_v2 = NaN(N,2,49);
EconomicIndicators_RF_v2 = NaN(N,2,4);
%
AUC_NM_ALasso=NaN(N,2);
PCC_NM_ALasso=NaN(N,2);
BS_NM_ALasso=NaN(N,2);
KS_NM_ALasso=NaN(N,2);
PGI_NM_ALasso=NaN(N,2);
RelmissClassifCost_NM_ALasso = NaN(N,2,49);
EconomicIndicators_NM_ALasso = NaN(N,2,4);
%
AUC_SVM=NaN(N,2);
PCC_SVM=NaN(N,2);
BS_SVM=NaN(N,2);
KS_SVM=NaN(N,2);
PGI_SVM=NaN(N,2);
RelmissClassifCost_SVM = NaN(N,2,49);
EconomicIndicators_SVM = NaN(N,2,4);
%
AUC_NN=NaN(N,2);
PCC_NN=NaN(N,2);
BS_NN=NaN(N,2);
KS_NN=NaN(N,2);
PGI_NN=NaN(N,2);
RelmissClassifCost_NN = NaN(N,2,49);
EconomicIndicators_NN = NaN(N,2,4);
%
N_Leaf_tot=NaN(N,2);
idxMin_tot=NaN(N,2);
depthM_tot=NaN(N,2);
%
N_Coeffs=NaN(N,2);
N_Coeffs_tot=NaN(N,2);
%
N_Coeffs_LR=NaN(N,2);
N_Coeffs_LR_tot=NaN(N,2);
N_Coeffs_NLR=NaN(N,2);
N_Coeffs_NLR_tot=NaN(N,2);
N_Coeffs_NLRLasso=NaN(N,2);
N_Coeffs_NLRLasso_tot=NaN(N,2);

%% Loop
for i=1:N
  for ii=1:2
    %% load data  
    if strcmp(whichdata,'Australian')
      [depLearning,depTest,explLearning,explTest,depLearningQual,explLearningLogistic,explTestLogistic, ...
          name_var_final,name_var_final_logistic]= importAustralianData(i,ii); 
    end    
    if strcmp(whichdata,'Housing')
      [depLearning,depTest,explLearning,explTest,depLearningQual,explLearningLogistic,explTestLogistic, ...
          name_var_final,name_var_final_logistic]= importHousingData(i,ii); 
    end
    if strcmp(whichdata,'Kaggle')
      [depLearning,depTest,explLearning,explTest,depLearningQual,explLearningLogistic,explTestLogistic, ...
          name_var_final,name_var_final_logistic]=importKaggleData(i,ii); 
    end
    if strcmp(whichdata,'Taiwan')
      [depLearning,depTest,explLearning,explTest,depLearningQual,explLearningLogistic,explTestLogistic, ...
          name_var_final,name_var_final_logistic]= importTaiwanData(i,ii); 
    end
    
    %%
    addpath(genpath('C:\Users\Mendeley\penalized'));
    
    %% logistic regression
    [beta,~,predictProbTest,predictClassTest,~,~]= ...
                       runLogistic(depLearning,explLearningLogistic,explTestLogistic,name_var_final_logistic); 
     [AUC_Logistic(i,ii),PCC_Logistic(i,ii),BS_Logistic(i,ii),...
                  KS_Logistic(i,ii),PGI_Logistic(i,ii),missClassifCost_Logistic,EconomicIndicators_Logistic(i,ii,:)]= ...
                  computeEvalCriteriaCS_Logistic(depTest,predictProbTest,predictClassTest);
     N_Coeffs_LR(i,ii)=sum(beta~=0);
     N_Coeffs_LR_tot(i,ii)=size(beta,1);
        
    %% Non Linear logistic regression
    [beta,~,predictProbTest,predictClassTest,~,~]= ...
                       runLogisticNL(depLearning,explLearningLogistic,explTestLogistic,explLearning,depLearningQual,name_var_final_logistic);
    [AUC_LogisticNL(i,ii),PCC_LogisticNL(i,ii),BS_LogisticNL(i,ii),...
                  KS_LogisticNL(i,ii),PGI_LogisticNL(i,ii),RelmissClassifCost_LogisticNL(i,ii,:),EconomicIndicators_LogisticNL(i,ii,:)]= ...
                  computeEvalCriteriaCS(depTest,predictProbTest,predictClassTest,missClassifCost_Logistic);
     N_Coeffs_NLR(i,ii)=sum(beta~=0);
     N_Coeffs_NLR_tot(i,ii)=size(beta,1);
             
    %% Non Linear logistic regression with Lasso
    output = runLogisticNLLasso(depLearning,explLearningLogistic,depTest,explTestLogistic,explLearning,...
                    depLearningQual); 
    [AUC_LogisticNL1(i,ii),PCC_LogisticNL1(i,ii),BS_LogisticNL1(i,ii),...
                  KS_LogisticNL1(i,ii),PGI_LogisticNL1(i,ii),RelmissClassifCost_LogisticNL1(i,ii,:),EconomicIndicators_LogisticNL1(i,ii,:)]= ...
                  computeEvalCriteriaCS(depTest,output.predict_alassoTest,output.binaire_alasso,missClassifCost_Logistic);
     N_Coeffs_NLRLasso(i,ii)=sum(output.Coeffs~=0);
     N_Coeffs_NLRLasso_tot(i,ii)=size(output.Coeffs,1);
              
      %% run random forest and get evaluation criteria values
      [~,predictProbTest,predictClassTest,N_Leaf,Nb_arbre,depthM]= ...
                             runRandomForest(depLearning,explLearning,explTest,depLearningQual);
       [AUC_RF_v2(i,ii),PCC_RF_v2(i,ii),BS_RF_v2(i,ii),...
                    KS_RF_v2(i,ii),PGI_RF_v2(i,ii),RelmissClassifCost_RF_v2(i,ii,:),EconomicIndicators_RF_v2(i,ii,:)]...
                  = computeEvalCriteriaCS(depTest,predictProbTest,predictClassTest,missClassifCost_Logistic);
       %       
       N_Leaf_tot(i,ii)=N_Leaf;  
       idxMin_tot(i,ii)=Nb_arbre;
       depthM_tot(i,ii)=depthM;  
       
      %% PLTR
      output=adaptivePenalizedLogisticTree2SplitsALasso(explLearning,depLearning,...
                                     explTest,depTest,depLearningQual,name_var_final);
      [AUC_NM_ALasso(i,ii),PCC_NM_ALasso(i,ii),BS_NM_ALasso(i,ii),...
       KS_NM_ALasso(i,ii),PGI_NM_ALasso(i,ii),RelmissClassifCost_NM_ALasso(i,ii,:),EconomicIndicators_NM_ALasso(i,ii,:)]...
           = computeEvalCriteriaCS(depTest,output.predict_alassoTest,output.binaire_alasso,missClassifCost_Logistic);
      % 
      N_Coeffs(i,ii)=sum(output.Coeffs~=0);  
      [N_Coeffs_tot(i,ii),~]=size(output.Coeffs);
      
      %% SVM
      [~,predictProbTest,predictClassTest,depTestSVM]= ...
                   runSVM(depLearning,explLearning,explTest,depLearningQual,depTest);
      [AUC_SVM(i,ii),PCC_SVM(i,ii),BS_SVM(i,ii),...
       KS_SVM(i,ii),PGI_SVM(i,ii),RelmissClassifCost_SVM(i,ii,:),EconomicIndicators_SVM(i,ii,:)]...
           = computeEvalCriteriaCS(depTestSVM,predictProbTest,predictClassTest,missClassifCost_Logistic);

      %% NN
      [~,predictProbTest,predictClassTest]= ...
                   runNN(depLearning,explLearning,explTest);
      [AUC_NN(i,ii),PCC_NN(i,ii),BS_NN(i,ii),...
       KS_NN(i,ii),PGI_NN(i,ii),RelmissClassifCost_NN(i,ii,:),EconomicIndicators_NN(i,ii,:)]...
           = computeEvalCriteriaCS(depTest,predictProbTest,predictClassTest,missClassifCost_Logistic);
      
  end
end

Results = [...
 [mean(mean(AUC_Logistic));mean(mean(AUC_LogisticNL));mean(mean(AUC_LogisticNL1));...
             mean(mean(AUC_RF_v2));mean(mean(AUC_NM_ALasso));mean(mean(AUC_SVM));mean(mean(AUC_NN))]...
 [mean(mean(PGI_Logistic));mean(mean(PGI_LogisticNL));mean(mean(PGI_LogisticNL1));...
            mean(mean(PGI_RF_v2));mean(mean(PGI_NM_ALasso));mean(mean(PGI_SVM));mean(mean(PGI_NN))]...
 [mean(mean(PCC_Logistic));mean(mean(PCC_LogisticNL));mean(mean(PCC_LogisticNL1));...
            mean(mean(PCC_RF_v2));mean(mean(PCC_NM_ALasso));mean(mean(PCC_SVM));mean(mean(PCC_NN))]...
 [mean(mean(KS_Logistic));mean(mean(KS_LogisticNL));mean(mean(KS_LogisticNL1));...
                mean(mean(KS_RF_v2));mean(mean(KS_NM_ALasso));mean(mean(KS_SVM));mean(mean(KS_NN))]... 
 [mean(mean(BS_Logistic));mean(mean(BS_LogisticNL));mean(mean(BS_LogisticNL1));...
                mean(mean(BS_RF_v2));mean(mean(BS_NM_ALasso));mean(mean(BS_SVM));mean(mean(BS_NN))]...
        ];  
          
output.Results=Results;
output.N_Coeffs = N_Coeffs;
output.N_Coeffs_tot=N_Coeffs_tot;
output.N_Coeffs_LR = N_Coeffs_LR;
output.N_Coeffs_LR_tot = N_Coeffs_LR_tot;
output.N_Coeffs_NLR = N_Coeffs_NLR;
output.N_Coeffs_NLR_tot = N_Coeffs_NLR_tot;
output.N_Coeffs_NLRLasso = N_Coeffs_NLRLasso;
output.N_Coeffs_NLRLasso_tot = N_Coeffs_NLRLasso_tot;
output.N_Leaf_tot=N_Leaf_tot;
output.idxMin_tot=idxMin_tot;
output.depthM_tot=depthM_tot;
output.RelmissClassifCost_LogisticNL=RelmissClassifCost_LogisticNL;
output.RelmissClassifCost_LogisticNL1=RelmissClassifCost_LogisticNL1;
output.RelmissClassifCost_RF_v2=RelmissClassifCost_RF_v2;
output.RelmissClassifCost_NM_ALasso=RelmissClassifCost_NM_ALasso;
output.RelmissClassifCost_SVM=RelmissClassifCost_SVM;
output.RelmissClassifCost_NN=RelmissClassifCost_NN;
output.EconomicIndicators_LR = EconomicIndicators_Logistic;
output.EconomicIndicators_NLR = EconomicIndicators_LogisticNL;
output.EconomicIndicators_NLRLasso = EconomicIndicators_LogisticNL1;
output.EconomicIndicators_RF = EconomicIndicators_RF_v2;
output.EconomicIndicators_PLTR = EconomicIndicators_NM_ALasso;
output.EconomicIndicators_SVM = EconomicIndicators_SVM;
output.EconomicIndicators_NN = EconomicIndicators_NN;
%
