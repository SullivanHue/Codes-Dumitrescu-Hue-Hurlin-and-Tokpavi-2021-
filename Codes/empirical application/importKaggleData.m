
function [depLearning,depTest,explLearning,explTest,depLearningQual,...
              explLearningLogistic,explTestLogistic,name_var_final,name_var_final_logistic]= importKaggleData(a,b)
           
data=load('C:\Users\p57904\Documents\Voie Recherche\Mémoire - 1er papier\dataset\1\kaggle.txt');
dep=data(:,2);
expl=data(:,3:end);
[~,KK]=size(expl);
for i=1:KK
   expl(isnan(expl(:,i)),i)=nanmean(expl(:,i));    
end
depLearningQual=[];
expl_logistic=expl; 


data_final=[expl dep];
data_final_logistic=[expl_logistic dep];
name_var_final={'RevolvingUtilizationOfUnsecuredLines', 'Age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', ...
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', ...
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'};
name_var_final_logistic=name_var_final;

[N_row_data_final,~]=size(data_final);

%% learning and test samples :2-fold
k=2;                                                                                                                           
rng(100000000+a);
s=rng;
rng(s);

kfold = cvpartition(N_row_data_final,'k',k);
for l=1:2               
index_in=kfold.training(l);                                               
index_out=kfold.test(l);                                                  
in_data_final=data_final(index_in,:);                                                   
out_data_final=data_final(index_out,:);
in_data_final_logistic=data_final_logistic(index_in,:);                                                   
out_data_final_logistic=data_final_logistic(index_out,:);
explLearning{l}=in_data_final(:,1:end-1);
explLearningLogistic{l}=in_data_final_logistic(:,1:end-1);
depLearning{l}=in_data_final(:,end);
explTest{l}=out_data_final(:,1:end-1);
explTestLogistic{l}=out_data_final_logistic(:,1:end-1);
depTest{l}=out_data_final(:,end);
end

explLearning=explLearning{b};
explLearningLogistic=explLearningLogistic{b};
depLearning=depLearning{b};
explTest=explTest{b};
explTestLogistic=explTestLogistic{b};
depTest=depTest{b};



