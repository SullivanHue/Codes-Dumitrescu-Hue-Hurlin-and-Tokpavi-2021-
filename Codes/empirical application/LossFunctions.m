function [LogVrais,BS] = LossFunctions(depTest,predictProbTest)

%% Brier's score
BS=(depTest-predictProbTest).^2;

%% Minus Logistic Log-Likelihood function
for ii = 1:size(predictProbTest,2);
for i = 1:size(predictProbTest,1);
if predictProbTest(i,ii) == 0; predictProbTest(i,ii) = 0.0000000001; end; %Pour transformer les proba égales à 0 pour ne pas que le log de la logvraisemblance produise une erreur
end;
end

LogVrais = depTest.*log(predictProbTest) + (1-depTest).*log(1-predictProbTest);

LogVrais = -LogVrais;

