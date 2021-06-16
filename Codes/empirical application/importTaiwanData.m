
function [depLearning,depTest,explLearning,explTest,depLearningQual,...
                explLearningLogistic,explTestLogistic,name_var_final,name_var_final_logistic] = importTaiwanData(a,b)

%% download data and recode the categorical predictor with many outcomes
[data,txt]=xlsread('C:\Users\p57904\Documents\Voie Recherche\Mémoire - 1er papier\dataset\1\taiwan_credit.xls');
dep=data(:,end);

%% get quantitative predictors & solve for missing values
depLearningQuant = [2 6 13:24];
expl = data(:,depLearningQuant); 
expl_logistic = expl;

%% get and manage the categorical predictors
depLearningQual = [3:5 7:12];
Q = data(:,depLearningQual); 
name_var_final_logistic = txt(2,depLearningQuant);
name_var_final = txt(2,2:end-1);
for i=1:size(Q,2)
 Qi = Q(:,i); 
 oo=tabulate(Qi);
 labeli = unique(Qi);
 varQ=zeros(size(Qi,1),1);
 for j=1:length(labeli)
   varQ=varQ+(Qi==labeli(j))*j;
 end
  expl = [expl varQ];
  dummY = dummyvar(varQ);
  expl_logistic = [expl_logistic dummY(:,1:end-1)];
end
depLearningQual = length(depLearningQuant)+1:size(expl,2);

%% final data
data_final=[expl dep];
data_final_logistic=[expl_logistic dep];
[N_row_data_final,~]=size(data_final);

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




