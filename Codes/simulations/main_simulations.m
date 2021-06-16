
clc
clear all

%% add paths
addpath(genpath('Z:\papers\Papier credit scoring\code\penalized'));
% install_penalized

%% parameters
nsim = 100;
P = 4:20;

%% output
pCC = NaN(nsim,length(P),6);
AUC = NaN(nsim,length(P),6);
BS = NaN(nsim,length(P),6);
PGI = NaN(nsim,length(P),6);
KS = NaN(nsim,length(P),6);
N_Leaf = NaN(nsim,1);
Nb_arbre = NaN(nsim,1);
depthM = NaN(nsim,1);
activeVarPLTR = NaN(nsim,1);

%% Loop
for sim=1:nsim
    sim
  for p=1:length(P)
     %simulate data 
     [explLearning,explTest,explLearningLogistic,explTestLogistic,depLearning,depTest,...
           depLearningQual,name_var_final,name_var_final_logistic_MARS] = DGP(sim,P(p));
     %run logistic regression
     [~,~,predictProbTest,predictClassTest,~,~]= ...
               runLogistic(depLearning,explLearningLogistic,explTestLogistic,name_var_final_logistic_MARS);
     [pCC(sim,p,1),AUC(sim,p,1),BS(sim,p,1),KS(sim,p,1),PGI(sim,p,1)] = computeEvalCriteria_Simu(depTest,predictClassTest,predictProbTest);
     %run non linear logistic regression
     [~,~,predictProbTest,predictClassTest,~,~]= ...
               runLogisticNL(depLearning,explLearningLogistic,explTestLogistic,explLearning,depLearningQual,name_var_final_logistic_MARS);
     [pCC(sim,p,2),AUC(sim,p,2),BS(sim,p,2),KS(sim,p,2),PGI(sim,p,2)] = computeEvalCriteria_Simu(depTest,predictClassTest,predictProbTest);
     % run random forest
     [~,predictProbTest,predictClassTest,N_Leaf(sim),Nb_arbre(sim),depthM(sim)]= ...
                   runRandomForest(depLearning,explLearning,explTest,depLearningQual);
     [pCC(sim,p,3),AUC(sim,p,3),BS(sim,p,3),KS(sim,p,3),PGI(sim,p,3)] = computeEvalCriteria_Simu(depTest,predictClassTest,predictProbTest);
     % run PLTR
     output=adaptivePenalizedLogisticTree2SplitsALasso(explLearning,depLearning,...
                             explTest,depTest,depLearningQual,name_var_final);
     [pCC(sim,p,4),AUC(sim,p,4),BS(sim,p,4),KS(sim,p,4),PGI(sim,p,4)] = computeEvalCriteria_Simu(depTest,output.binaire_alasso,output.predict_alassoTest);  
     activeVarPLTR(sim,1) = output.activeVar;
  end       
end
%
figure
mpCC = squeeze(mean(pCC));
mpCC = mpCC(1:end,:);
plot([4:P(end)]',mpCC,'LineWidth',1.5),
legend('Linear Logistic regression','Non-Linear Logistic regression','Random Forest','PLTR','location','best','FontSize',11),
xlabel('Number of Predictors')
ylabel('Proportion of Correct Classification (PCC)')
grid on

figure
mAUC = squeeze(mean(AUC));
mAUC = mAUC(1:end,:);
plot([4:P(end)]',mAUC,'LineWidth',1.5),
legend('Linear Logistic regression','Non-Linear Logistic regression','Random Forest','PLTR','location','best','FontSize',11),
xlabel('Number of Predictors')
ylabel('Area Under the ROC Curve (AUC)')
grid on

figure
mBS = squeeze(mean(BS));
mBS = mBS(1:end,:);
plot([4:P(end)]',mBS,'LineWidth',1.5),
legend('Linear Logistic regression','Non-Linear Logistic regression','Random Forest','PLTR','location','best','FontSize',11),
xlabel('Number of Predictors')
ylabel('Brier score (BS)')
grid on
