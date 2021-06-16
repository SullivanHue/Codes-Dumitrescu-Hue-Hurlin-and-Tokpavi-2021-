
function [explLearning,explTest,explLearningLogistic,explTestLogistic,depLearning,depTest,...
    depLearningQual,name_var_final,name_var_final_logistic_MARS] = DGP(N_sim,p)

N=5000; %Taille de l'échantillon

rng(123456789+N_sim); %seed

x=normrnd(0,1,[N,p]); %predictive variables
x2=sort(x);x3=x2(501:4500,:);

beta0=unifrnd(-1,1,1,1); %constante
beta1=unifrnd(-1,1,p,1); %betas singletons
beta2=unifrnd(-1,1,((p*(p-1))/2),1); %betas couples

gamma=[]; %thresholds gamma
for j=1:p
    g=x3(randi(length(x3)),j);
    gamma=[gamma;g];
end

delta=[]; %thresholds delta
for j=1:p
    d=x3(randi(length(x3)),j);
    delta=[delta;d];
end

xi=[]; %couples of interactions
for j=1:p-1
    for k=j+1:p
        xx=(x(:,j)<=delta(j)).*x(:,k)<=delta(k);
        xi=[xi xx];
    end
end

index=beta0+(x<=repmat(gamma',size(x,1),1))*beta1+xi*beta2; %index function

Proba=1./(1+exp(-index)); %Probabilities

pi=median(Proba); %Thresholds Pi

y=double((Proba>pi)); %Y : Default or not

k=2;                                                                                                                           
kfold = cvpartition(N,'k',k);
in_Training=kfold.training(1);  %Indices échantillon training
in_Test=kfold.test(1); %Indices échantillon test

explLearning=x(in_Training,:); %Echantillon Training expl
explTest=x(in_Test,:); %Echantillon Test expl
explLearningLogistic=x(in_Training,:); %Echantillon Training expl
explTestLogistic=x(in_Test,:); %Echantillon Test expl

depLearning=y(in_Training,:); %Echantillon Training dep
depTest=y(in_Test,:);%Echantillon Test dep

depLearningQual=[];

name_var_final=num2cell([1:p]); %Noms des variables
name_var_final_logistic_MARS=num2cell([1:p]); %Noms des variables


