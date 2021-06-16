clc
clear

% ROC : AUC2, AUC1; results = percentage of cases where model 2 and model 1 do not have similar forecasting abilities (the one with higher AUC is better)
% DM :  two-sided case results = percentage of cases where model 2 and model 1 do not have similar forecasting abilities (the one with lower loss is better)
% DM : if one-sided test then: d=loss1-loss2 >0 => 2nd model better (one-sided test to the right) ; results = percentage of cases where model 2 is better than model 1  
% MCS : results = percentage of cases where model i (i in 1,2,3) is in MCS90 / MCS75


sig = 0.05; % significance level
sigm = 0.1; % MCS alpha  => MCS90%
sigm2 = 0.25; % MCS alpha  => MCS75%

%% Housing


load('outputHousing.mat')

for i = 1:10
    
    Hat_Proba = [outputHousing.PredictProb_LR(:,i) outputHousing.PredictProb_RF(:,i) outputHousing.PredictProb_PLTR(:,i)];
 
 % set probabilities between 0 and 1 excluded   
    for k = 1:size(Hat_Proba,1)
        for l = 1:size(Hat_Proba,2)
        
            if Hat_Proba(k,l)==1
            
            Hat_Proba(k,l)=0.99999999999999; 
         
         end
         
         
         if Hat_Proba(k,l)==0
            
            Hat_Proba(k,l)=0.00000000000000001;
            
         end
        end
    end

    
    
    Y = outputHousing.DepTest_LR(:,i);
    
    [LogVrais,BS] = LossFunctions(Y,Hat_Proba);
    
    [All_ROC,DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais);
    
    output.DM_BS = DM_BS;
    output.DM_LogVrais = DM_LogVrais;
    output.table_BS = table_BS;
    output.table_LogVrais = table_LogVrais;
    output.All_ROC = All_ROC;
    
    %To save
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceHousing','_',A); 
    save(name,'output'); 
    
end

% Table rejection percentages

Table_DM_BS = zeros(1,3);
Table_DM_LogVrais = zeros(1,3);
Table_MCS_BS = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_MCS_LogVrais = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_All_ROC =zeros(1,3);

for i = 1:10
    
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceHousing','_',A,'.mat');     
    load(name)
    
    Table_DM_BS = Table_DM_BS + [(output.DM_BS(1,2)<sig) (output.DM_BS(2,2)<sig) (output.DM_BS(3,2)<sig)];
    Table_DM_LogVrais = Table_DM_LogVrais + [(output.DM_LogVrais(1,2)<sig) (output.DM_LogVrais(2,2)<sig) (output.DM_LogVrais(3,2)<sig)];
    Table_All_ROC = Table_All_ROC + [(output.All_ROC(1,4)<sig) (output.All_ROC(2,4)<sig) (output.All_ROC(3,4)<sig)];
    Table_MCS_BS = Table_MCS_BS + [(output.table_BS(1,2)>sigm) (output.table_BS(2,2)>sigm) (output.table_BS(3,2)>sigm); (output.table_BS(1,2)>sigm2) (output.table_BS(2,2)>sigm2) (output.table_BS(3,2)>sigm2)];
    Table_MCS_LogVrais = Table_MCS_LogVrais + [(output.table_LogVrais(1,2)>sigm) (output.table_LogVrais(2,2)>sigm) (output.table_LogVrais(3,2)>sigm); (output.table_LogVrais(1,2)>sigm2) (output.table_LogVrais(2,2)>sigm2) (output.table_LogVrais(3,2)>sigm2)];

end

Table_DM_BS_Housing = (Table_DM_BS./10).*100;
Table_DM_LogVrais_Housing = (Table_DM_LogVrais./10).*100;
Table_All_ROC_Housing = (Table_All_ROC./10).*100;
Table_MCS_BS_Housing = (Table_MCS_BS./10).*100;
Table_MCS_LogVrais_Housing = (Table_MCS_LogVrais./10).*100;
%%

clc
clear

%% Australian
sig = 0.05; % significance level
sigm = 0.1; % MCS alpha  => MCS90%
sigm2 = 0.25; % MCS alpha  => MCS75%

load('outputAustralian.mat')

for i = 1:10
    
    Hat_Proba = [outputAustralian.PredictProb_LR(:,i) outputAustralian.PredictProb_RF(:,i) outputAustralian.PredictProb_PLTR(:,i)];
    
    % set probabilities between 0 and 1 excluded   
    for k = 1:size(Hat_Proba,1)
        for l = 1:size(Hat_Proba,2)
        
            if Hat_Proba(k,l)==1
            
            Hat_Proba(k,l)=0.99999999999999; 
         
         end
         
         
         if Hat_Proba(k,l)==0
            
            Hat_Proba(k,l)=0.00000000000000001;
            
         end
        end
    end
    
    Y = outputAustralian.DepTest_LR(:,i);
    
    [LogVrais,BS] = LossFunctions(Y,Hat_Proba);
    
%    [DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais);
     [All_ROC,DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais);
    
    output.DM_BS = DM_BS;
    output.DM_LogVrais = DM_LogVrais;
    output.table_BS = table_BS;
    output.table_LogVrais = table_LogVrais;
    output.All_ROC = All_ROC;
    
    %To save
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceAustralian','_',A); 
    save(name,'output'); 
    
end

% Table rejection percentages

Table_DM_BS = zeros(1,3);
Table_DM_LogVrais = zeros(1,3);
Table_MCS_BS = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_MCS_LogVrais = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_All_ROC =zeros(1,3);

for i = 1:10
    
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceAustralian','_',A,'.mat');     
    load(name)
    
    Table_DM_BS = Table_DM_BS + [(output.DM_BS(1,2)<sig) (output.DM_BS(2,2)<sig) (output.DM_BS(3,2)<sig)];
    Table_DM_LogVrais = Table_DM_LogVrais + [(output.DM_LogVrais(1,2)<sig) (output.DM_LogVrais(2,2)<sig) (output.DM_LogVrais(3,2)<sig)];
    Table_All_ROC = Table_All_ROC + [(output.All_ROC(1,4)<sig) (output.All_ROC(2,4)<sig) (output.All_ROC(3,4)<sig)];
    Table_MCS_BS = Table_MCS_BS + [(output.table_BS(1,2)>sigm) (output.table_BS(2,2)>sigm) (output.table_BS(3,2)>sigm); (output.table_BS(1,2)>sigm2) (output.table_BS(2,2)>sigm2) (output.table_BS(3,2)>sigm2)];
    Table_MCS_LogVrais = Table_MCS_LogVrais + [(output.table_LogVrais(1,2)>sigm) (output.table_LogVrais(2,2)>sigm) (output.table_LogVrais(3,2)>sigm); (output.table_LogVrais(1,2)>sigm2) (output.table_LogVrais(2,2)>sigm2) (output.table_LogVrais(3,2)>sigm2)];
    
end

Table_DM_BS_Australian = (Table_DM_BS./10).*100;
Table_DM_LogVrais_Australian = (Table_DM_LogVrais./10).*100;
Table_All_ROC_Australian = (Table_All_ROC./10).*100;
Table_MCS_BS_Australian = (Table_MCS_BS./10).*100;
Table_MCS_LogVrais_Australian = (Table_MCS_LogVrais./10).*100;

%%

clc
clear

%% Kaggle
sig = 0.05; % significance level
sigm = 0.1; % MCS alpha  => MCS90%
sigm2 = 0.25; % MCS alpha  => MCS75%

load('outputKaggle.mat')

for i =5 % 5:10
    
    Hat_Proba = [outputKaggle.PredictProb_LR(:,i) outputKaggle.PredictProb_RF(:,i) outputKaggle.PredictProb_PLTR(:,i)];
    
    % set probabilities between 0 and 1 excluded   
    for k = 1:size(Hat_Proba,1)
        for l = 1:size(Hat_Proba,2)
        
            if Hat_Proba(k,l)==1
            
            Hat_Proba(k,l)=0.99999999999999; 
         
         end
         
         
         if Hat_Proba(k,l)==0
            
            Hat_Proba(k,l)=0.00000000000000001;
            
         end
        end
    end
    
    Y = outputKaggle.DepTest_LR(:,i);
    
    [LogVrais,BS] = LossFunctions(Y,Hat_Proba);
    
    %[DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais);
    [All_ROC,DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais);
    
    output.DM_BS = DM_BS;
    output.DM_LogVrais = DM_LogVrais;
    output.table_BS = table_BS;
    output.table_LogVrais = table_LogVrais;
    output.All_ROC = All_ROC;
    
    %To save
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceKaggle','_',A); 
    save(name,'output'); 
    
end

% Table rejection percentages

Table_DM_BS = zeros(1,3);
Table_DM_LogVrais = zeros(1,3);
Table_MCS_BS = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_MCS_LogVrais = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_All_ROC =zeros(1,3);

for i = 1:10
    
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceKaggle','_',A,'.mat');     
    load(name)
    
    Table_DM_BS = Table_DM_BS + [(output.DM_BS(1,2)<sig) (output.DM_BS(2,2)<sig) (output.DM_BS(3,2)<sig)];
    Table_DM_LogVrais = Table_DM_LogVrais + [(output.DM_LogVrais(1,2)<sig) (output.DM_LogVrais(2,2)<sig) (output.DM_LogVrais(3,2)<sig)];
    Table_All_ROC = Table_All_ROC + [(output.All_ROC(1,4)<sig) (output.All_ROC(2,4)<sig) (output.All_ROC(3,4)<sig)];
    Table_MCS_BS = Table_MCS_BS + [(output.table_BS(1,2)>sigm) (output.table_BS(2,2)>sigm) (output.table_BS(3,2)>sigm); (output.table_BS(1,2)>sigm2) (output.table_BS(2,2)>sigm2) (output.table_BS(3,2)>sigm2)];
    Table_MCS_LogVrais = Table_MCS_LogVrais + [(output.table_LogVrais(1,2)>sigm) (output.table_LogVrais(2,2)>sigm) (output.table_LogVrais(3,2)>sigm); (output.table_LogVrais(1,2)>sigm2) (output.table_LogVrais(2,2)>sigm2) (output.table_LogVrais(3,2)>sigm2)];
    
end

Table_DM_BS_Kaggle = (Table_DM_BS./10).*100;
Table_DM_LogVrais_Kaggle = (Table_DM_LogVrais./10).*100;
Table_All_ROC_Kaggle = (Table_All_ROC./10).*100;
Table_MCS_BS_Kaggle = (Table_MCS_BS./10).*100;
Table_MCS_LogVrais_Kaggle = (Table_MCS_LogVrais./10).*100;

%%

clc
clear

%% Taiwan
sig = 0.05; % significance level
sigm = 0.1; % MCS alpha  => MCS90%
sigm2 = 0.25; % MCS alpha  => MCS75%

load('outputTaiwan.mat')

for i = 6:7
    
    Hat_Proba = [outputTaiwan.PredictProb_LR(:,i) outputTaiwan.PredictProb_RF(:,i) outputTaiwan.PredictProb_PLTR(:,i)];
    
    % set probabilities between 0 and 1 excluded   
    for k = 1:size(Hat_Proba,1)
        for l = 1:size(Hat_Proba,2)
        
            if Hat_Proba(k,l)==1
            
            Hat_Proba(k,l)=0.99999999999999; 
         
         end
         
         
         if Hat_Proba(k,l)==0
            
            Hat_Proba(k,l)=0.00000000000000001;
            
         end
        end
    end
    
    Y = outputTaiwan.DepTest_LR(:,i);
    
    [LogVrais,BS] = LossFunctions(Y,Hat_Proba);
    
%   [DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais);
    [All_ROC,DM_BS,DM_LogVrais,table_BS,table_LogVrais] = RunInference(Y,Hat_Proba,BS,LogVrais);
    
    output.DM_BS = DM_BS;
    output.DM_LogVrais = DM_LogVrais;
    output.table_BS = table_BS;
    output.table_LogVrais = table_LogVrais;
    output.All_ROC = All_ROC;
    
    %To save
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceTaiwan','_',A); 
    save(name,'output'); 
    
end

% Table rejection percentages

Table_DM_BS = zeros(1,3);
Table_DM_LogVrais = zeros(1,3);
Table_MCS_BS = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_MCS_LogVrais = zeros(2,3); % one row for pval>1-0.9 and one for pval>1-0.75
Table_All_ROC =zeros(1,3);

for i = 1:10
    
    A = sprintf('%1$g',i);
    name = strcat('outputInferenceTaiwan','_',A,'.mat');     
    load(name)
    
    Table_DM_BS = Table_DM_BS + [(output.DM_BS(1,2)<sig) (output.DM_BS(2,2)<sig) (output.DM_BS(3,2)<sig)];
    Table_DM_LogVrais = Table_DM_LogVrais + [(output.DM_LogVrais(1,2)<sig) (output.DM_LogVrais(2,2)<sig) (output.DM_LogVrais(3,2)<sig)];
    Table_All_ROC = Table_All_ROC + [(output.All_ROC(1,4)<sig) (output.All_ROC(2,4)<sig) (output.All_ROC(3,4)<sig)];
      Table_MCS_BS = Table_MCS_BS + [(output.table_BS(1,2)>sigm) (output.table_BS(2,2)>sigm) (output.table_BS(3,2)>sigm); (output.table_BS(1,2)>sigm2) (output.table_BS(2,2)>sigm2) (output.table_BS(3,2)>sigm2)];
    Table_MCS_LogVrais = Table_MCS_LogVrais + [(output.table_LogVrais(1,2)>sigm) (output.table_LogVrais(2,2)>sigm) (output.table_LogVrais(3,2)>sigm); (output.table_LogVrais(1,2)>sigm2) (output.table_LogVrais(2,2)>sigm2) (output.table_LogVrais(3,2)>sigm2)];
   
end

Table_DM_BS_Taiwan = (Table_DM_BS./10).*100;
Table_DM_LogVrais_Taiwan = (Table_DM_LogVrais./10).*100;
Table_All_ROC_Taiwan = (Table_All_ROC./10).*100;
Table_MCS_BS_Taiwan = (Table_MCS_BS./10).*100;
Table_MCS_LogVrais_Taiwan = (Table_MCS_LogVrais./10).*100;

