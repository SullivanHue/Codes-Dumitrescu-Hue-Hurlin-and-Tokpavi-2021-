%Appel ROC test
% input crisis proba1 proba2 step (proba1:model1 ; proba2:model2)
% construct sensit specif for each model to compute AUC1 and AUC2
% compute the variance-covariance matrix of (AUC1 AUC2)

%{

crisis=OOScrisis(:,2);
proba1=OOSProba_stat(:,2);
proba2=OOSProba(:,2);

%}

function[Wauc, Wpvalue, AUC1, AUC2]=AppelROC(crisis, proba1, proba2)

%Compute AUCs
%====================================
y1=[crisis proba1];
y2=[crisis proba2];
step=0.001;                                                                % Step in the cut-off change
%step=0.01;                                                                % Step in the cut-off change


[R1]= sens_spec1(y1, step);

[R2]= sens_spec1(y2, step);

AUC1=trapz(R1.specif, R1.sensit);                     % The area under the ROC curve
AUC2=trapz(R2.specif, R2.sensit);                     % The area under the ROC curve

if AUC1~=AUC2
    %compute the vcv matrix
    %====================================
    
    %a. Kernel (model1)
    
    m=sum(crisis);                                                    % number of crisis periods
    n=length(crisis)-m;                                              % number of calm periods
    
    X1=crisis.*proba1;
    f1=find(X1);
    X11=X1(f1);                                                        %keep proba for the crisis periods (Correct order!! -given by the time-periods)
    
    Y1=(1-crisis).*proba1;
    g1=find(Y1);
    Y11=Y1(g1);                                                        %keep proba for the calm periods (Correct order!!)
    
    
    for i=1:m
        for j=1:n
            
            if Y11(j)<X11(i)
                K1(i,j)=1;
            elseif Y11(j)==X11(i)
                K1(i,j)=1/2;
            else
                K1(i,j)=0;
            end
        end
    end
    
    %a. Kernel (model2)
    
    m=sum(crisis);                                                    % number of crisis periods
    n=length(crisis)-m;                                              % number of calm periods
    
    X2=crisis.*proba2;
    f2=find(X2);
    X22=X2(f2);                                                        %keep proba for the crisis periods (Correct order!!)
    
    Y2=(1-crisis).*proba2;
    g2=find(Y2);
    Y22=Y2(g2);                                                        %keep proba for the calm periods (Correct order!!)
    
    
    
    for i=1:m
        for j=1:n
            
            if Y22(j)<X22(i)
                K2(i,j)=1;
            elseif Y22(j)==X22(i)
                K2(i,j)=1/2;
            else
                K2(i,j)=0;
                
            end
        end
    end
    
    %Elements
    
    Va10=mean(K1,2);
    Va01=mean(K1);
    
    Vb10=mean(K2,2);
    Vb01=mean(K2);
    
    
    %verifier AUC
    %auc1=sum(sum(K1),2)/(m*n);
    %auc2=sum(sum(K2),2)/(m*n);
    
    %b. First component (S1: 2*2 matrix)
    s10aa=sum ( (Va10-AUC1).*(Va10-AUC1) )/(m-1);
    
    s10ab=sum ( (Va10-AUC1).*(Vb10-AUC2) )/(m-1);
    
    s10ba=sum ( (Va10-AUC1).*(Vb10-AUC2) )/(m-1);
    
    s10bb=sum ( (Vb10-AUC2).*(Vb10-AUC2) )/(m-1);
    
    S10=[s10aa s10ab; s10ba s10bb];
    
    %c. Second component (S2: 2*2 matrix)
    
    s01aa=sum ( (Va01-AUC1).*(Va01-AUC1) )/(n-1);
    
    s01ab=sum ( (Va01-AUC1).*(Vb01-AUC2) )/(n-1);
    
    s01ba=sum ( (Va01-AUC1).*(Vb01-AUC2) )/(n-1);
    
    s01bb=sum ( (Vb01-AUC2).*(Vb01-AUC2) )/(n-1);
    
    S01=[s01aa s01ab; s01ba s01bb];
    
    %d. VCV=S1/n+S2/m (2*2 matrix)
    
    Vcv=S10/m+S01/n;                                             % Variance-covariance matrix
    
    %Stat du test
    
    Wauc=(AUC1-AUC2)^2/(Vcv(1,1)+Vcv(2,2)-2*Vcv(1,2));% test-statistic
    
    Wpvalue=1-chi2cdf(Wauc,1);                              % we compare 2 series => DF=1
    
    
else
    Wauc=0;                                                      %chi2inv(0,1); observe values >0 in all 100%cases
    Wpvalue=1;
end

%{
dd=[Va10 Vb10];
drr=( dd'*dd-m*[AUC1 ;AUC2]*[AUC1 AUC2]  )/(m-1);
%}


