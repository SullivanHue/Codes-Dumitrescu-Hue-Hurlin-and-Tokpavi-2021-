
function [depLearning,depTest,explLearning,explTest,depLearningQual,...
               explLearningLogistic,explTestLogistic,name_var_final,name_var_final_logistic] = importAustralianData(a,b)
           
data=load('C:\Users\p57904\Documents\Voie Recherche\Mémoire - 1er papier\dataset\1\australian.dat');
dep=data(:,end);
expl=data(:,1:end-1);
[~,KK]=size(expl);


depLearningQuant=[2 3 7 10 13 14];
for i=1:KK
   if ismember(i,depLearningQuant)
     expl(isnan(expl(:,i)),i)=nanmean(expl(:,i));    
   end
end

%% specific treatment for logistic regression
depLearningQual=[1 4 5 6 8 9 11 12];
varQ=[];
for i=1:length(depLearningQual)
    if min(expl(:,depLearningQual(:,i)))==0
      ephem=expl(:,depLearningQual(:,i))+1;
      dum=dummyvar(ephem); 
      varQ=[varQ dum(:,1:end-1)];      
    else
      dum=dummyvar(expl(:,depLearningQual(:,i))); 
      varQ=[varQ dum(:,1:end-1)];
    end
end
expl_logistic=[expl(:,depLearningQuant) varQ];

%% final data
data_final=[expl dep];
name_var_final={'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14'};
data_final_logistic=[expl_logistic dep];
name_var_final_logistic={'A2','A3','A7','A10','A13','A14',...
                                   'a',...
                                    'p','g',...
                                    'ff','d','i','k','j','aa','m','c','w','e','q','r','cc',...
                                    'ff','dd','j','bb','v','n','o','h',...
                                    't',...
                                    't',...
                                    't',...
                                    's','g',...
                                    };
                                    
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






