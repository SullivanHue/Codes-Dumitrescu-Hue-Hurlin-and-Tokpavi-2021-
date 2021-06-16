
clc
clear all

%% add paths
addpath(genpath('C:\Users\Mendeley\penalized'));
install_penalized

%% parameters
N = 5;

%% Kaggle Dataset
outputKaggle = mainSingle(N,'Kaggle');
save outputKaggle outputKaggle
ResultsKaggle = outputKaggle.Results;


%% Taiwan Dataset
outputTaiwan = mainSingle(N,'Taiwan');
save outputTaiwan outputTaiwan
ResultsTaiwan = outputTaiwan.Results;


%% Australian Dataset
outputAustralian = mainSingle(N,'Australian');
save outputAustralian outputAustralian
ResultsAustralian = outputAustralian.Results;
%

%% Housing Dataset
outputHousing = mainSingle(N,'Housing');
save outputHousing outputHousing
ResultsHousing = outputHousing.Results;

%% FigEcoKaggle
% Data
load('outputKaggle')
NLLr = squeeze(mean(mean(outputKaggle.RelmissClassifCost_LogisticNL)));
NLLrALasso = squeeze(mean(mean(outputKaggle.RelmissClassifCost_LogisticNL1)));
RF = squeeze(mean(mean(outputKaggle.RelmissClassifCost_RF_v2)));
PLTR = squeeze(mean(mean(outputKaggle.RelmissClassifCost_NM_ALasso)));
SVM = squeeze(mean(mean(outputKaggle.RelmissClassifCost_SVM)));
NN = squeeze(mean(mean(outputKaggle.RelmissClassifCost_NN)));
% Fig
figure(1),plot(NLLr,'LineWidth',1.5,'Color',[0.8500 0.3250 0.0980])
hold on
plot(NLLrALasso,'LineWidth',1.5,'Color',[0.4660 0.6740 0.1880])
plot(RF,'LineWidth',1.5,'Color',[0.9290 0.6940 0.1250])
plot(PLTR,'LineWidth',1.5,'Color',[0.4940 0.1840 0.5560])
plot(SVM,'LineWidth',1.5,'Color',[0.3010 0.7450 0.9330])
plot(NN,'LineWidth',1.5,'Color',[0 0 0])
legend('Non Linear Logistic regression','Non Linear Logistic regression+ALasso','Random Forest','PLTR','Support Vector Machine','Neural Network','Location','best','FontSize',11)
xlabel('C_{FN}')
ylabel('Cost reduction (in %)')
grid on
%% FigEcoTaiwan
% Data
load('outputTaiwan')
NLLr = squeeze(mean(mean(outputTaiwan.RelmissClassifCost_LogisticNL)));
NLLrALasso = squeeze(mean(mean(outputTaiwan.RelmissClassifCost_LogisticNL1)));
RF = squeeze(mean(mean(outputTaiwan.RelmissClassifCost_RF_v2)));
PLTR = squeeze(mean(mean(outputTaiwan.RelmissClassifCost_NM_ALasso)));
SVM = squeeze(mean(mean(outputTaiwan.RelmissClassifCost_SVM)));
NN = squeeze(mean(mean(outputTaiwan.RelmissClassifCost_NN)));
% Fig
figure(1),plot(NLLr,'LineWidth',1.5,'Color',[0.8500 0.3250 0.0980])
hold on
plot(NLLrALasso,'LineWidth',1.5,'Color',[0.4660 0.6740 0.1880])
plot(RF,'LineWidth',1.5,'Color',[0.9290 0.6940 0.1250])
plot(PLTR,'LineWidth',1.5,'Color',[0.4940 0.1840 0.5560])
plot(SVM,'LineWidth',1.5,'Color',[0.3010 0.7450 0.9330])
plot(NN,'LineWidth',1.5,'Color',[0 0 0])
legend('Non Linear Logistic regression','Non Linear Logistic regression+ALasso','Random Forest','PLTR','Support Vector Machine','Neural Network','Location','best','FontSize',11)
xlabel('C_{FN}')
ylabel('Cost reduction (in %)')
grid on
%% FigEcoAustralian
% Data
load('outputAustralian')
NLLr = squeeze(mean(mean(outputAustralian.RelmissClassifCost_LogisticNL)));
NLLrALasso = squeeze(mean(mean(outputAustralian.RelmissClassifCost_LogisticNL1)));
RF = squeeze(mean(mean(outputAustralian.RelmissClassifCost_RF_v2)));
PLTR = squeeze(mean(mean(outputAustralian.RelmissClassifCost_NM_ALasso)));
SVM = squeeze(mean(mean(outputAustralian.RelmissClassifCost_SVM)));
NN = squeeze(mean(mean(outputAustralian.RelmissClassifCost_NN)));
% Fig
figure(1),plot(NLLr,'LineWidth',1.5,'Color',[0.8500 0.3250 0.0980])
hold on
plot(NLLrALasso,'LineWidth',1.5,'Color',[0.4660 0.6740 0.1880])
plot(RF,'LineWidth',1.5,'Color',[0.9290 0.6940 0.1250])
plot(PLTR,'LineWidth',1.5,'Color',[0.4940 0.1840 0.5560])
plot(SVM,'LineWidth',1.5,'Color',[0.3010 0.7450 0.9330])
plot(NN,'LineWidth',1.5,'Color',[0 0 0])
legend('Non Linear Logistic regression','Non Linear Logistic regression+ALasso','Random Forest','PLTR','Support Vector Machine','Neural Network','Location','best','FontSize',11)
xlabel('C_{FN}')
ylabel('Cost reduction (in %)')
grid on
%% FigEcoHousing
% Data
load('outputHousing')
NLLr = squeeze(mean(mean(outputHousing.RelmissClassifCost_LogisticNL)));
NLLrALasso = squeeze(mean(mean(outputHousing.RelmissClassifCost_LogisticNL1)));
RF = squeeze(mean(mean(outputHousing.RelmissClassifCost_RF_v2)));
PLTR = squeeze(mean(mean(outputHousing.RelmissClassifCost_NM_ALasso)));
SVM = squeeze(mean(mean(outputHousing.RelmissClassifCost_SVM)));
NN = squeeze(mean(mean(outputHousing.RelmissClassifCost_NN)));
% Fig
figure(1),plot(NLLr,'LineWidth',1.5,'Color',[0.8500 0.3250 0.0980])
hold on
plot(NLLrALasso,'LineWidth',1.5,'Color',[0.4660 0.6740 0.1880])
plot(RF,'LineWidth',1.5,'Color',[0.9290 0.6940 0.1250])
plot(PLTR,'LineWidth',1.5,'Color',[0.4940 0.1840 0.5560])
plot(SVM,'LineWidth',1.5,'Color',[0.3010 0.7450 0.9330])
plot(NN,'LineWidth',1.5,'Color',[0 0 0])
legend('Non Linear Logistic regression','Non Linear Logistic regression+ALasso','Random Forest','PLTR','Support Vector Machine','Neural Network','Location','best','FontSize',11)
xlabel('C_{FN}')
ylabel('Cost reduction (in %)')
grid on