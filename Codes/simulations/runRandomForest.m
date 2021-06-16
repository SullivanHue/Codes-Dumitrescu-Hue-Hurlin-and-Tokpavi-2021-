
function [predictProbLearning,predictProbTest,predictClassTest,N_Leaf,Nb_arbre,depthM]= ...
                   runRandomForest(depLearning,explLearning,explTest,depLearningQual)

%% fix random number generator for reproducibility
rng(200000);
s=rng;
rng(s);

%% find the optimal number of trees
Prior_Prob=[1-sum((depLearning==1))/length(depLearning) sum((depLearning==1))/length(depLearning)];
nTrees_set=200;
Train=TreeBagger(nTrees_set,explLearning,depLearning,'OOBPred','on','OOBVarImp','on',...
                                           'CategoricalPredictors',depLearningQual,...
                                           'MinLeafSize',5,'Prior',Prior_Prob);
oobErrorBaggedEnsemble = oobError(Train);
nbTree_opt=min(find(oobErrorBaggedEnsemble==min(oobErrorBaggedEnsemble)));


%% Random forest with the optimal number of trees
b = TreeBagger(nbTree_opt,explLearning,depLearning,'OOBPred','on','OOBVarImp','on',...
                               'CategoricalPredictors',depLearningQual,...
                               'MinLeafSize',5,'Prior',Prior_Prob);
                           
%Calcul du nombre de feuilles terminales
N_Leaf=0;
for i=1:nbTree_opt
    N_Leaf=N_Leaf+sum(sum(b.Trees{i}.Children==0));
end;
Nb_arbre=nbTree_opt;
N_Leaf=N_Leaf/Nb_arbre;

%Calcul de la profondeur de l'arbre
nodeT=[];
for i=1:Nb_arbre
 parent = b.Trees{i}.Parent;
 depth = 0;
 node = parent(end);
 while node~=0
        depth = depth + 1;
        node = parent(node);
 end
 nodeT=[nodeT;depth];
end
depthM=mean(nodeT);

%% predictions
[~,classifScore] = b.predict(explLearning);
predictProbLearning=classifScore(:,2);                          
nbEvent = sum(depLearning==1);
oo = 0:0.001:1;
nbo = NaN(length(oo),1);
for i=1:length(oo)
  nbo(i)=sum(predictProbLearning>=oo(i));    
end
[~,idx]=min(abs(nbo-nbEvent));
OptCO_RF = oo(idx);
%
[~,classifScore] = b.predict(explTest);
predictProbTest=classifScore(:,2);
predictClassTest = (predictProbTest>=OptCO_RF);
predictClassTest=double(predictClassTest);




