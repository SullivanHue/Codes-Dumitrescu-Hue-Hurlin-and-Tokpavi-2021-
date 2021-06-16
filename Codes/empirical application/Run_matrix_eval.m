%Run_evaluation_tests - VCV MATRIX -
%--------------------------------------
% creates matrices /  computes losses / evaluate-MCS
% E Dumitrescu
%-------------------
clc
clear

addpath (genpath('/Users/edumitrescu/Dropbox/CES/Ch Bilel/CODURI_NOI'));
addpath(genpath('/Users/Denisa/Dropbox/CES/Ch Bilel/CODURI_NOI/'));   
addpath(genpath('/Users/Denisa/Dropbox/CES/Ch Bilel/CODURI_NOI/cDCC'));

casess=5;

% define cc as one of the cases!!!!!
%cc=1;  % CJPM sample1
%cc=3; % BPG sample1


%cc=2; % CJPM sample2
%cc=4; % BPG sample2
cc=5; % PEPKO sample2 


%cDCC all
load cDCCfcst_ALL % cDCCfcstBACC1 cDCCfcstBACC2 cDCCfcstBPG1 cDCCfcstBPG2 cDCCfcstcJPM1 cDCCfcstCJPM2 cDCCfcstPEPKO2; Columns 2:4 v1,v2,cov in each ;all forecasts from cDCC, for both pairs of assets and both periods
    
if cc==1    

% C-JPM sample1
load Res_oos_GJR_CJPM_sample1 % GJR_fcst

load Res_oos_RG_CJPM_sample1 %  RG_fcst

load Res_oos_RRG_CJPM_sample1 % RRG_fcst

%bivariate RG
load ForecastBvRG_CJPM1 % bfcst   - biv fcst v1 v2 cor ; RK0h1 RK1h1 RKCov1 RV0h1 RV1h1 RVCov1 - true OOS proxi RK and RV

%cDCC
cDCC=cDCCfcstBACC1;

elseif cc==2
    
% C-JPM sample2

load Res_oos_GJR_CJPM_sample2 % GJR_fcst

load Res_oos_RG_CJPM_sample2 %  RG_fcst

load Res_oos_RRG_CJPM_sample2 % RRG_fcst

%load Res_oos_GJR_GAS_CJPM_sample2 % GJR_fcst


%bivariate RG
load ForecastBvRG_CJPM2 % bfcst   - biv fcst v1 v2 cor ; RK0h1 RK1h1 RKCov1 RV0h1 RV1h1 RVCov1 - true OOS proxi RK and RV

%cDCC
cDCC=cDCCfcstBACC2;

elseif cc==3

% B-PG sample1

load Res_oos_GJR_BPG_sample1 % GJR_fcst

load Res_oos_RG_BPG_sample1 %  RG_fcst

load Res_oos_RRG_BPG_sample1 % RRG_fcst

%bivariate RG
load ForecastBvRG_BPG1 % bfcst   - biv fcst v1 v2 cor ; RK0h1 RK1h1 RKCov1 RV0h1 RV1h1 RVCov1 - true OOS proxi RK and RV

%cDCC
cDCC=cDCCfcstBPG1;

elseif cc==4

% B-PG sample2

load Res_oos_GJR_BPG_sample2 % GJR_fcst

load Res_oos_RG_BPG_sample2 %  RG_fcst

load Res_oos_RRG_BPG_sample2 % RRG_fcst

%load Res_oos_GJR_GAS_BPG_sample2 % GJR_fcst

%bivariate RG
load ForecastBvRG_BPG2 % bfcst   - biv fcst v1 v2 cor ; RK0h1 RK1h1 RKCov1 RV0h1 RV1h1 RVCov1 - true OOS proxi RK and RV

%cDCC
cDCC=cDCCfcstBPG2;

elseif cc==5
    
    
 % PEP-KO sample2

load Res_oos_GJR_GRAS_PEPKO_sample2 % GJR_fcst

load Res_oos_RG_PEPKO_sample2 %  RG_fcst

load Res_oos_RRG_PEPKO_sample2 % RRG_fcst

% load Res_oos_GJR_GAS_PEPKO_sample2 % GJR_fcst

%bivariate RG
load ForecastBvRG_PEPKO2 % bfcst   - biv fcst v1 v2 cor ; RK0h1 RK1h1 RKCov1 RV0h1 RV1h1 RVCov1 - true OOS proxi RK and RV

%cDCC
cDCC=cDCCfcstPEPKO2;    
    
end




%%

%--------------------------------
% Frobenius norm - matrix loss
%--------------------------------
nof= size(RK0h1,1);                                                            % no of forecasts

Pk_h1=NaN*ones(2,2,nof);                                                   % RK OOS all covariance matrices

Prv_h1=Pk_h1;                                                              % RV OOS all covariance matrices

Hrg_h1=Pk_h1;                                                              % RG 

Hrrg_h1=Pk_h1;                                                             % RRG

Hgjr_h1=Pk_h1;                                                             % GJR

HcDCC_h1=Pk_h1;                                                            % cDCC

Hbiv_h1=Pk_h1;                                                             % Biv RG


for i=1:nof
%kernel
    Pk_h1(:,:,i)=[RK0h1(i,1) RKCov1(i,1) ;...
                  RKCov1(i,1)  RK1h1(i,1)];
              
  %RV
    Prv_h1(:,:,i)=[RV0h1(i,1) RVCov1(i,1)  ;...
                  RVCov1(i,1) RV1h1(i,1)];
              
  %RG
      Hrg_h1(:,:,i)=[RG_fcst(i,1) RG_fcst(i,3) ;...
                  RG_fcst(i,3) RG_fcst(i,2) ];
  %RRG         
              
     Hrrg_h1(:,:,i)=[RRG_fcst(i,1) RRG_fcst(i,3) ;...
                  RRG_fcst(i,3) RRG_fcst(i,2) ];
                               
  %GJR
    Hgjr_h1(:,:,i)=[GJR_fcst(i,1) GJR_fcst(i,3) ;...
                  GJR_fcst(i,3) GJR_fcst(i,2) ];
              
  %Bivariate
    Hbiv_h1(:,:,i)=[bfcst(i,1) bfcst(i,3)*sqrt(bfcst(i,1)*bfcst(i,2))  ;...
                  bfcst(i,3)*sqrt(bfcst(i,1)*bfcst(i,2)) bfcst(i,2) ];
                  
   %cDCC           
              
     HcDCC_h1(:,:,i)=[cDCC(i,2) cDCC(i,4);...
                  cDCC(i,4) cDCC(i,3) ];                                                       
         
               %CCC                        
    % Hrg_h1(:,:,i)=[CCC.H0(i,1) CCC.Corr(i,1)*sqrt(CCC.H0(i,1)*CCC.H1(i,1) )  ;...
    %              CCC.Corr(i,1)*sqrt(CCC.H0(i,1)*CCC.H1(i,1) ) CCC.H1(i,1) ];
              
      
end          
nrm=5;          % number of models compared

%compute losses
Loss_k1=NaN*ones(nof,nrm);

Loss_rv1=NaN*ones(nof,nrm);


for i=1:nof

    % all horizons, proxi = R kernel
    
Loss_k1(i,1)=sum(eig((  Pk_h1(:,:,i) - Hbiv_h1(:,:,i)).^2));

Loss_k1(i,2)=sum(eig((  Pk_h1(:,:,i) -HcDCC_h1(:,:,i)).^2));

Loss_k1(i,3)=sum(eig((  Pk_h1(:,:,i) -Hrrg_h1(:,:,i)).^2));

Loss_k1(i,4)=sum(eig((  Pk_h1(:,:,i) -Hrg_h1(:,:,i)).^2));

Loss_k1(i,5)=sum(eig((  Pk_h1(:,:,i) -Hgjr_h1(:,:,i)).^2));

% all horizons, proxi = RV
    
Loss_rv1(i,1)=sum(eig((  Prv_h1(:,:,i) - Hbiv_h1(:,:,i)).^2));

Loss_rv1(i,2)=sum(eig((  Prv_h1(:,:,i) -HcDCC_h1(:,:,i)).^2));

Loss_rv1(i,3)=sum(eig((  Prv_h1(:,:,i) -Hrrg_h1(:,:,i)).^2));

Loss_rv1(i,4)=sum(eig((  Prv_h1(:,:,i) -Hrg_h1(:,:,i)).^2));

Loss_rv1(i,5)=sum(eig((  Prv_h1(:,:,i) - Hgjr_h1(:,:,i)).^2));

end


%--------------------------------
% MCS test
%--------------------------------


alpha=0.1;
B= 10000;            %no replications
w=12;                % block length 1 week
boot='block';       % block bootstrap
addpath(genpath('/Users/edumitrescu/MEGA/PAPERS/ES_backtest/sheppard_mfe_toolbox'));
addpath(genpath('/Users/Denisa/Dropbox/CES/Ch Bilel/CODURI_NOI/sheppard_mfe_toolbox'));

%RK
%pval order: excluded, included
lossesF=Loss_k1;
[includedRkF,pvalsRFk,excludedRkF,includedSQkF,pvalsSQkF,excludedSQkF]=mcs(lossesF,alpha,B,w,boot);
A=[(mean(lossesF))' [excludedSQkF;includedSQkF]  pvalsSQkF];


   all=[excludedSQkF; includedSQkF];
   pval=[pvalsSQkF all];
   nht=length(all); %nb of competing models
   zz=(1:nht)';
   rank_raw1=NaN*ones(nht,2);
   
for i=1:nht
    rank_raw1(i,2)=pval(find(zz(i)==pval(:,2)),1);
 
end

table_forb_rk=[mean(lossesF)' rank_raw1(:,2)];


%RV
lossesF=Loss_rv1;
[includedRvF,pvalsRFv,excludedRvF,includedSQvF,pvalsSQvF,excludedSQvF]=mcs(lossesF,alpha,B,w,boot);
A2=[(mean(lossesF))' [excludedSQvF;includedSQvF]  pvalsSQvF];
A=[A;A2];

 %all=[excludedRvF; includedRvF];
 %  pval=[pvalsRFv all];
   all=[excludedSQvF; includedSQvF];
   pval=[pvalsSQvF all];
   nht=length(all); %nb of competing models
   zz=(1:nht)';
   rank_raw1=NaN*ones(nht,2);
   
for i=1:nht
    rank_raw1(i,2)=pval(find(zz(i)==pval(:,2)),1);
 
end

table_forb_rv=[mean(lossesF)' rank_raw1(:,2)];



%Multivariate QLike loss

%compute losses
Loss_k1=NaN*ones(nof,nrm);
Loss_rv1=NaN*ones(nof,nrm);

for i=1:nof

    % all horizons, proxi = R kernel
    
Loss_k1(i,1)=trace( pinv(Hbiv_h1(:,:,i))*Pk_h1(:,:,i)) -log(det(pinv(Hbiv_h1(:,:,i))*Pk_h1(:,:,i) ))- 2;

Loss_k1(i,2)=trace( pinv(HcDCC_h1(:,:,i))*Pk_h1(:,:,i)) -log(det(pinv(HcDCC_h1(:,:,i))*Pk_h1(:,:,i) ))- 2;

Loss_k1(i,3)=trace( pinv(Hrrg_h1(:,:,i))*Pk_h1(:,:,i)) -log(det(pinv(Hrrg_h1(:,:,i))*Pk_h1(:,:,i) ))- 2;

Loss_k1(i,4)=trace( pinv(Hrg_h1(:,:,i))*Pk_h1(:,:,i)) -log(det(pinv(Hrg_h1(:,:,i))*Pk_h1(:,:,i) ))- 2;

Loss_k1(i,5)=trace( pinv(Hgjr_h1(:,:,i))*Pk_h1(:,:,i)) -log(det(pinv(Hgjr_h1(:,:,i))*Pk_h1(:,:,i) ))- 2;

% all horizons, proxi = RV
  
Loss_rv1(i,1)=trace( pinv(Hbiv_h1(:,:,i))*Prv_h1(:,:,i)) -log(det(pinv(Hbiv_h1(:,:,i))*Prv_h1(:,:,i) ))- 2;

Loss_rv1(i,2)=trace( pinv(HcDCC_h1(:,:,i))*Prv_h1(:,:,i)) -log(det(pinv(HcDCC_h1(:,:,i))*Prv_h1(:,:,i) ))- 2;

Loss_rv1(i,3)=trace( pinv(Hrrg_h1(:,:,i))*Prv_h1(:,:,i)) -log(det(pinv(Hrrg_h1(:,:,i))*Prv_h1(:,:,i) ))- 2;

Loss_rv1(i,4)=trace( pinv(Hrg_h1(:,:,i))*Prv_h1(:,:,i)) -log(det(pinv(Hrg_h1(:,:,i))*Prv_h1(:,:,i) ))- 2;

Loss_rv1(i,5)=trace( pinv(Hgjr_h1(:,:,i))*Prv_h1(:,:,i)) -log(det(pinv(Hgjr_h1(:,:,i))*Prv_h1(:,:,i) ))- 2;

end


%--------------------------------
% MCS test
%--------------------------------


alpha=0.1;
B= 10000;            %no replications
w=12;                % block length 1 week
boot='block';       % block bootstrap
addpath '/Users/eidumitrescu/Dropbox/ExchR_codes/Evaluation_may/Evaluation/bootstrap';
addpath '/Users/ivona_d//Dropbox/ExchR_codes/Evaluation_may/Evaluation/bootstrap';

%pval order: excluded, included
lossesQ=Loss_k1;
[includedRk,pvalsRk,excludedRk,includedSQk,pvalsSQk,excludedSQk]=mcs(lossesQ,alpha,B,w,boot);
D=[(mean(lossesQ))' [excludedSQk;includedSQk]  pvalsSQk];

 all=[excludedSQk; includedSQk];
   pval=[pvalsSQk all];
 %all=[excludedRk; includedRk];
 %  pval=[pvalsRk all];
   nht=length(all); %nb of competing models
   zz=(1:nht)';
   rank_raw1=NaN*ones(nht,2);
   
for i=1:nht
    rank_raw1(i,2)=pval(find(zz(i)==pval(:,2)),1);
 
end

table_q_rk=[mean(lossesQ)' rank_raw1(:,2)];


%RV
lossesQ=Loss_rv1;
[includedRv,pvalsRv,excludedRv,includedSQv,pvalsSQv,excludedSQv]=mcs(lossesQ,alpha,B,w,boot);
D2=[(mean(lossesQ))' [excludedSQv;includedSQv]  pvalsSQv];
D=[D;D2];

 all=[excludedSQv; includedSQv];
   pval=[pvalsSQv all];
   nht=length(all); %nb of competing models
   zz=(1:nht)';
   rank_raw1=NaN*ones(nht,2);
   
for i=1:nht
    rank_raw1(i,2)=pval(find(zz(i)==pval(:,2)),1);
 
end

table_q_rv=[mean(lossesQ)' rank_raw1(:,2)];



%entrywise norm

%%%%%%%%%%%%%%%%%
%Entrywise 1 norm

for i=1:nof
       % all horizons, proxi = R kernel
    
Loss_k1(i,1)=sum(sum(abs( Pk_h1(:,:,i) - Hbiv_h1(:,:,i))));

Loss_k1(i,2)=sum(sum(abs(  Pk_h1(:,:,i) -HcDCC_h1(:,:,i))));

Loss_k1(i,3)=sum(sum(abs( Pk_h1(:,:,i) - Hrrg_h1(:,:,i))));

Loss_k1(i,4)=sum(sum(abs(  Pk_h1(:,:,i) -Hrg_h1(:,:,i))));

Loss_k1(i,5)=sum(sum(abs(  Pk_h1(:,:,i) -Hgjr_h1(:,:,i))));


% all horizons, proxi = RV
    
Loss_rv1(i,1)=sum(sum(abs(  Prv_h1(:,:,i) - Hbiv_h1(:,:,i))));

Loss_rv1(i,2)=sum(sum(abs(  Prv_h1(:,:,i) -HcDCC_h1(:,:,i))));

Loss_rv1(i,3)=sum(sum(abs(  Prv_h1(:,:,i) - Hrrg_h1(:,:,i))));

Loss_rv1(i,4)=sum(sum(abs(  Prv_h1(:,:,i) -Hrg_h1(:,:,i))));   

Loss_rv1(i,5)=sum(sum(abs(  Prv_h1(:,:,i) -Hgjr_h1(:,:,i))));   

end


%--------------------------------
% MCS test
%--------------------------------


alpha=0.1;
B= 10000;            %no replications
w=12;                % block length 1 week
boot='block';       % block bootstrap
addpath '/Users/eidumitrescu/Dropbox/ExchR_codes/Evaluation_may/Evaluation/bootstrap';
addpath '/Users/ivona_d//Dropbox/ExchR_codes/Evaluation_may/Evaluation/bootstrap';

%pval order: excluded, included
losses=Loss_k1;
[includedRE,pvalsRE,excludedRE,includedSQE,pvalsSQE,excludedSQE]=mcs(losses,alpha,B,w,boot);
C=[(mean(losses))' [excludedSQE;includedSQE]  pvalsSQE];


 all=[excludedSQE; includedSQE];
   pval=[pvalsSQE all];
   nht=length(all); %nb of competing models
   zz=(1:nht)';
   rank_raw1=NaN*ones(nht,2);
   
for i=1:nht
    rank_raw1(i,2)=pval(find(zz(i)==pval(:,2)),1);
 
end

table_e_rk=[mean(losses)' rank_raw1(:,2)];



lossesvE=Loss_rv1;
[includedRvE,pvalsRvE,excludedRvE,includedSQvE,pvalsSQvE,excludedSQvE]=mcs(losses,alpha,B,w,boot);
C2=[(mean(lossesvE))' [excludedSQvE;includedSQvE]  pvalsSQvE];
C=[C;C2];

all=[excludedSQvE; includedSQvE];
   pval=[pvalsSQvE all];
   nht=length(all); %nb of competing models
   zz=(1:nht)';
   rank_raw1=NaN*ones(nht,2);
   
for i=1:nht
    rank_raw1(i,2)=pval(find(zz(i)==pval(:,2)),1);
 
end

table_e_rv=[mean(lossesvE)' rank_raw1(:,2)];



%{

figure(1)
subplot(2,2,1)
semilogy(sqrt(250*[OOStrue.RV2(:,1) OOSboot.H2(:,1)  OOSgarch.H2(:,1) ]) )
subplot(2,2,2)
semilogy(sqrt(250*[OOStrue.RV2(:,2) OOSboot.H2(:,2)  OOSgarch.H2(:,2) ]) )
subplot(2,2,3)
semilogy(sqrt(250*[OOStrue.RV2(:,3) OOSboot.H2(:,3)  OOSgarch.H2(:,3) ]) )
subplot(2,2,4)
semilogy(sqrt(250*[OOStrue.RV2(:,4) OOSboot.H2(:,4)  OOSgarch.H2(:,4) ]) )

figure(2)
subplot(2,2,1)
plot(sqrt(250*[OOStrue.RV2(:,1) OOSboot.H2(:,1)  OOSgarch.H2(:,1) ]) )
subplot(2,2,2)
plot(sqrt(250*[OOStrue.RV2(:,2) OOSboot.H2(:,2)  OOSgarch.H2(:,2) ]) )
subplot(2,2,3)
plot(sqrt(250*[OOStrue.RV2(:,3) OOSboot.H2(:,3)  OOSgarch.H2(:,3) ]) )
subplot(2,2,4)
plot(sqrt(250*[OOStrue.RV2(:,4) OOSboot.H2(:,4)  OOSgarch.H2(:,4) ]) )


figure(1)
subplot(2,2,1)
plot([OOStrue.RCorr(:,1) OOSboot.Corr(:,1)  cDCC.Corr(:,1) ] )
subplot(2,2,2)
plot([OOStrue.RCorr(:,2) OOSboot.Corr(:,2)  cDCC.Corr(:,2) ] )
subplot(2,2,3)
plot([OOStrue.RCorr(:,3) OOSboot.Corr(:,3)  cDCC.Corr(:,3) ] )
subplot(2,2,4)
plot([OOStrue.RCorr(:,4) OOSboot.Corr(:,4)  cDCC.Corr(:,4) ] )

%}
