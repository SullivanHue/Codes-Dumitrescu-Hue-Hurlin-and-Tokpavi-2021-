function [All_ROC, DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais)

%%
%--------------------------------
% 1. ROC - AUC test
%--------------------------------

% define Hat_Proba as the matrix of probabilities (T observations * K columns, for T individuals and K competing models)
%( first column is the benchmark model (e.g. logistic reg), then the other models )

% define Y the column vector of 0-1 dependent variable 


%initialize
Wauc = NaN*ones(size(Hat_Proba,2)-1,1);
Wpvalue = NaN*ones(size(Hat_Proba,2)-1,1);
AUC1 = NaN*Wpvalue;
AUC2 = NaN*Wpvalue;


%table of results
%----------------------------------------------------

Wauc = NaN(3,1); Wpvalue = NaN(3,1); AUC1 = NaN(3,1); AUC2 = NaN(3,1);

[Wauc(1,1), Wpvalue(1,1), AUC1(1,1), AUC2(1,1)] = AppelROC(Y, Hat_Proba(:,1), Hat_Proba(:,2)); % first: model of possible interest
[Wauc(2,1), Wpvalue(2,1), AUC1(2,1), AUC2(2,1)] = AppelROC(Y, Hat_Proba(:,1), Hat_Proba(:,3)); % first: model of possible interest
[Wauc(3,1), Wpvalue(3,1), AUC1(3,1), AUC2(3,1)] = AppelROC(Y, Hat_Proba(:,2), Hat_Proba(:,3)); % first: model of possible interest

 All_ROC = [AUC2 AUC1 Wauc Wpvalue]; % AUC ROC Test results

%% 
%--------------------------------
% 2. Diebold-Mariano test
%--------------------------------


% compute / load matrix with vectors of losses (T observations * K columns, for T individuals and K competing models)
% for eg a loss function is : (Y - Hat_Proba)^2, with Y taking values 0 or 1 (the dependent variable) and 
% Hat_Proba the estimated probability (vector)
% first column, K=1, for the benchmark model (e.g. linear reg), then the others

%the other loss function they wanted is minus-log-likelihood, if you have
%it for all models.

h = 0;

DM_1 = NaN(3,1); DM_pval_1 = NaN(3,1);
[DM_1(1,1), DM_pval_1(1,1)]  = dmtest(BS(:,1), BS(:,2), h, 0);% DM Test results, loss function 1
[DM_1(2,1), DM_pval_1(2,1)]  = dmtest(BS(:,1), BS(:,3), h, 0);% DM Test results, loss function 1
[DM_1(3,1), DM_pval_1(3,1)]  = dmtest(BS(:,2), BS(:,3), h, 0);% DM Test results, loss function 1
DM_BS = [DM_1 DM_pval_1];

DM_2 = NaN(3,1); DM_pval_2 = NaN(3,1);
[DM_2(1,1), DM_pval_2(1,1)]  = dmtest(LogVrais(:,1), LogVrais(:,2), h,0 );% DM Test results, loss function 2
[DM_2(2,1), DM_pval_2(2,1)]  = dmtest(LogVrais(:,1), LogVrais(:,3), h,0 );% DM Test results, loss function 2
[DM_2(3,1), DM_pval_2(3,1)]  = dmtest(LogVrais(:,2), LogVrais(:,3), h,0 );% DM Test results, loss function 2
DM_LogVrais = [DM_2 DM_pval_2];

%%
%--------------------------------
% 3. MCS test - all models at once
%--------------------------------


alpha = 0.1;
B = 10000;            %no replications
w = 12;                % block length 1 week
boot = 'block';       % block bootstrap
addpath(genpath('C:\Users\Mendeley\mfe-toolbox-master'));

% loss function 1

[includedRk,pvalsRk,excludedRk,includedSQk,pvalsSQk,excludedSQk] = mcs(BS,alpha,B,w,boot);
A1 = [(mean(BS))' [excludedSQk;includedSQk]  pvalsSQk]; %pval order: excluded, included

   all = [excludedSQk; includedSQk];
   pval = [pvalsSQk all];
   nht = length(all); % nb of competing models
   zz = (1:nht)';
   rank_raw1 = NaN*ones(nht,2);
   
for i = 1:nht
    rank_raw1(i,2) = pval(find(zz(i)==pval(:,2)),1);
 
end

table_BS = [mean(BS)' rank_raw1(:,2)]; 

% loss function 2

[includedRk,pvalsRk,excludedRk,includedSQk,pvalsSQk,excludedSQk] = mcs(LogVrais,alpha,B,w,boot);
A2 = [(mean(LogVrais))' [excludedSQk;includedSQk]  pvalsSQk]; %pval order: excluded, included

   all = [excludedSQk; includedSQk];
   pval = [pvalsSQk all];
   nht = length(all); % nb of competing models
   zz = (1:nht)';
   rank_raw2 = NaN*ones(nht,2);
   
for i = 1:nht
    rank_raw2(i,2) = pval(find(zz(i)==pval(:,2)),1);
 
end

table_LogVrais = [mean(LogVrais)' rank_raw2(:,2)]; 



