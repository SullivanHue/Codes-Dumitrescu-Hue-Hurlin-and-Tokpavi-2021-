% RUN ROC test
% on OOS static (1) and dynamic (2) logit models 
%-----------------------------------------------
%input
% table of proba1: N columns for all N countries
% table of proba2: N columns for all N countries
% table of crises: N columns for all N countries
%-----------------------------------------------
%output
% ROC test statistic and p-value
%-----------------------------------------------

clc

clear

% load results for static logit model
%----------------------------------------------------    
    %load OOSRESULTS_static             % Rolling OOS with mres
    %load OOSRESULTSfixed_static        % Fixed OOS with mres
    %load OOSRESULTSfixed_static_b      % Fixed OOS with mreg
    
    load OOSRESULTS_6m_static            % Rolling OOS with mreg

%check if there are crises in OOS
Ocr=sum(OOScrisis);                         % number of crisis periods in OOS for each country
POcr=find(Ocr==0);                          % identify the countries that do not exhibit crises in OOS

%----------------------------------------------------
%keep only the countries that do know crises in OOS
%----------------------------------------------------

OOSProba_stat=OOSProba;                     % take the matrix of probabilities

OOSProba_stat(:,POcr)=[];                   % delete the countries without crises in OOS


clear OOSProba OOScrisis

% load results for dynamic logit model (y_t-1)
%----------------------------------------------------
    %load OOSRESULTS
    % load OOSRESULTSfixed
    % load OOSRESULTSfixed_b
    load OOSRESULTS_6m
    
OOSProba(:,POcr)=[];                        % delete the countries without crises in OOS

OOScrisis(:,POcr)=[];                       % delete the countries without crises in OOS


%initialize
Wauc=NaN*ones(size(OOSProba,2),1);
Wpvalue=NaN*ones(size(OOSProba,2),1);
AUC1=NaN*Wpvalue;
AUC2=NaN*Wpvalue;

%replace NaN (non-convergence) with previous value
%----------------------------------------------------

OOSProba(1,15)=1;  %first value is NaN
OOSProba_stat (1,15)=1;  %first value is NaN

for j=1:size(OOSProba,2)
    
    for i=1:size(OOSProba,1)

        if isnan(OOSProba(i,j))==1
            
            OOSProba(i,j)=OOSProba(i-1,j);
            
        end
        
         if isnan(OOSProba_stat(i,j))==1
            
            OOSProba_stat(i,j)=OOSProba_stat(i-1,j);
            
         end
        
        
    % Replace proba=1 or 0 with 0.999999999999999and 0.0000000000000001
         
         if OOSProba(i,j)==1
            
            OOSProba(i,j)=0.99999999999999;
            
        end
        
         if OOSProba_stat(i,j)==1
            
            OOSProba_stat(i,j)=0.99999999999999; 
         
         end
         
         
         if OOSProba(i,j)==0
            
            OOSProba(i,j)=0.00000000000000001;
            
        end
        
         if OOSProba_stat(i,j)==0
            
            OOSProba_stat(i,j)=0.00000000000000001;
            
         end
         
         
         
    end
    
end


%table of results
%----------------------------------------------------

for i=1:size(OOSProba,2)

[Wauc(i), Wpvalue(i), AUC1(i), AUC2(i)]=AppelROC(OOScrisis(:,i), OOSProba(:,i), OOSProba_stat(:,i)); % first:dynamic

end

 A=[AUC2 AUC1 Wauc Wpvalue];











