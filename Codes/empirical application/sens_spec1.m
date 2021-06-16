%==========================================================================
% ==== Sensitivity and Specificity Computation for a certain country ====
%==========================================================================
% function usage: [R]= sens_spec(y, step)
%--------------------------------------------------------------------------
% INPUT:
% - y: dataset -1st column - observed crisis series;
%              -2nd column - probability series;
% - step: defines the cutoff values (and thus the rolling computation of
% sensitivity and specificity; it is recommended to choose a value between 
% 0.0001 and 0.01
%--------------------------------------------------------------------------
% OUTPUT:
%R.sensit: Sensitivity
%R.specif: Specificity
%R.ind  :  Cutoff values
%R.J    :  Younden Index
%R.TA   :  Total accuracy
%R.MCC  :  MCC
%R.TME  :  Total misclassification error
%R.WME  :  Weighted misclassification error
%==========================================================================
% Elena Dumitrescu
% LEO, 01.05.2011
%==========================================================================



function [R]= sens_spec1(y, step)

e=size(y,1);                                                               % Number of observations

indic=0;                                                                   % Initialization of a counting variable

ind=(0:step:1)';                                                           % Vector of cut-off values 

sensit=ones(length(ind),1)*NaN;                                            % Initialization of the vectors of results

specif=ones(length(ind),1)*NaN;

tac=ones(length(ind),1)*NaN;

MCC=ones(length(ind),1)*NaN;

TME=ones(length(ind),1)*NaN;

WME=ones(length(ind),1)*NaN;

for i=0:step:1;                                                            % For each cut-off c in [0,1]
    
    indic=indic+1;                                                         % Counting the elements in the vector of results
    
    x=ones(e,1);                                                           % Vector of ones
    
    xx=x.*i ;                                                              % Vector taking the value of the cut-off  
    
    a=zeros(e,1);                                                          % Intialization of useful vectors: predicted events;

    TP=zeros(e,1);                                                         % TP vector;
    
    FN=zeros(e,1);                                                         % FN vector;
    
    FP=zeros(e,1);                                                         % FP vector;
    
    TN=zeros(e,1);                                                         % TN vector.
    
    for j=1:e
        
        if y(j,2)>xx(j)
            
            a(j,1)=1;                                                      % Vector of predicted events (1 if crisis, 0 if not)
            
            if y(j,1)==1
                
                TP(j,1)=1;
                
            else
                
                FP(j,1)=1 ;                                                    % FP value
            end
            
        else
            
            a(j,1)=0;
            
            if y(j,1)==1
                
                FN(j,1)=1    ;
                
            else
                
                TN(j,1)=1 ;
            end
            
        end                                                              % End of the if condition
        
        
        %         if (y(j,1)==1) &&(a(j,1)==1);
        %
        %             TP(j,1)=1;                                                     % TP value
        %
        %         elseif (y(j,1)==1)&& (a(j,1)==0);
        %
        %             FN(j,1)=1    ;                                                 % FN value
        %
        %         elseif (y(j,1)==0) && (a(j,1)==1);
        %
        %             FP(j,1)=1 ;                                                    % FP value
        %
        %         elseif (y(j,1)==0) && (a(j,1)==0);
        %
        %             TN(j,1)=1 ;                                                    % TN value
        %
        %         end;                                                               % End of the if condition
        
    end                                                                   % End of the loop on j
    
       tpr=sum(TP) ;                                                       % The four elements of the contingency table for a certain value of the cut-off
    
       fnr=sum(FN);
       
       fpr=sum(FP);
       
       tnr=sum(TN);
       
%---------------------------------------------       
%sensitivity and specificity for each cut-off
%---------------------------------------------       

       sensit(indic)=tpr/(tpr+fnr);                                        % Vector of Sensitivity for all cut-offs
 
       specif(indic)=tnr/(fpr+tnr);                                        % Vector of Specificity for all cut-offs    
      
       if specif(indic)==Inf
       
           specif(indic)=1;                                                % Taking care of extreme situations
          
       else 
       
           specif(indic)=specif(indic);
   
       end
%-------------------
% Accuracy measures
%-------------------       

%        tac(indic)=(tpr+tnr)./e;                                            % Total accuracy measure    
%            
%        if sqrt( (tpr+fnr)*(tpr+fpr)*(tnr+fpr)*(tnr+fnr))~=0
%            
%            MCC(indic)=((tpr*tnr)-(fpr*fnr))/ sqrt( (tpr+fnr)*(tpr+fpr)*(tnr+fpr)*(tnr+fnr)); % MCC measure
%            
%        else
%            
%            MCC(indic)=0;
%            
%        end                                                                 % end of the if statement
% %----------------------------------
% % Misclassification error measures      
% %----------------------------------
%        
%    %=-Total errors
%        
%        LFN = 1/(tpr + fnr);
%        
%        LFP = 1/(tnr + fpr);
%        
%        TLN = fnr*LFN;
%        
%        TLP = fpr*LFP;
%        
%        TME(indic) = TLN+TLP;                                               % Total error
%         
%    %=-weighted errors
%        
%        WFN = LFN/(LFN+LFP);
%         
%        WFP = LFP/(LFN+LFP);
%        
%        WLN = (fnr*WFN)/e;
%        
%        WPN = (fpr*WFP)/e;
%        
%        WME(indic)= WLN+WPN;                                                % Weighted error
%        
%        
end          
%     
 %---------
 % Results
 %---------
 
 R.sensit=sensit;                                                          % Sensitivity
 
 R.specif=specif;                                                          % Specificity
 
 %R.gg=sensit-specif;

%  R.ind=ind;                                                                % Cutoff values
%   
%  R.J=sensit+specif-1;                                                      % Younden Index
%    
%  R.TA=tac;                                                                 % Total accuracy
%  
%  R.MCC=MCC;                                                                % MCC
%  
%  R.TME=TME;                                                                % Total misclassification error
%  
%  R.WME=WME;                                                                % Weighted misclassification error
%  
 
 







