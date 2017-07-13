function [ latex2 ] = numbers_experiments_rewhistogram( experimentcellS, experimentcellM, column1str, numtrials, tablename, tablelabel, threshold, optimalP, optimalM )
    clc;
    CONFIDENCE = 0.9;

    offset1 = optimalP;
    offset2 = optimalM;
    
    
    
    input.data = cell(length(experimentcellS), 6);
    input.tableColLabels = {'mean','percent good','mean, stdv of good','mean','percent good','mean, stdv of good'};
    input.tableRowLabels = {};
    extramsg = '' ;
    %% first pass to determine winners
    values = zeros(6,length(experimentcellS),numtrials);
    best = zeros(6,length(experimentcellS));
    for dbstrat = 1:length(experimentcellS)
        % SWINGUP
        dbs = experimentcellS{dbstrat}; 
        experiment = dbs{2};
        rewtrajs = load_experiment_results(experiment);
        if (size(rewtrajs,1) < numtrials) 
            extramsg = [' Based on at least ' num2str(size(rewtrajs,1)) ' experiments'];
        end
        rewtrajs = rewtrajs(1:min(numtrials,size(rewtrajs,1)),end-999:end);
        
        values(1,dbstrat,:) = mean(rewtrajs,2);
        values(2,dbstrat,:) = (sum(rewtrajs.' > -0.4)/10); %based on 1000 samples
        values(3,dbstrat,:) = max(rewtrajs.') - offset1;
        
        % MAGMAN
        dbs = experimentcellM{dbstrat}; 
        experiment = dbs{2};
        rewtrajs = load_experiment_results(experiment);
        if (size(rewtrajs,1) < numtrials) 
            extramsg = [' Based on at least ' num2str(size(rewtrajs,1)) ' experiments'];
        end
        rewtrajs = rewtrajs(1:min(numtrials,size(rewtrajs,1)),end-999:end);
        
        values(4,dbstrat,:) = mean(rewtrajs,2);
        values(5,dbstrat,:) = (sum(rewtrajs.' > -0.4)/10); %based on 1000 samples
        values(6,dbstrat,:) = max(rewtrajs.') - offset2;
        
    end
    
    for cat = 1:6 
        [~, idx] = max(mean(squeeze(values(cat,:,:)),2));
        for exp = 1:size(values,2)
            temp = anova1([squeeze(values(cat,idx,:)) squeeze(values(cat,exp,:))],[],'off') ;
            if temp >= (1-CONFIDENCE)
                best(cat,exp) = 0.5;
            end
            if temp == 1
                best(cat,exp) = 1;
            end
        end
    end
        
         
         
         
%         sum         = length(rewtrajs(:));
%         goodones    = rewtrajs(rewtrajs > threshold);
%         goodp       = 100 * (length(goodones)/sum);
%         goodmean    = mean(goodones);
%         goodstd     = std(goodones);
%         
%         name        = dbsname;
%     
%         
%         input.data{dbstrat,1} = num2str(mean(rewtrajs(:)) , '%.2f' );
%         input.data{dbstrat,2} = num2str(goodp, '%.1f \\%%' );
%         input.data{dbstrat,3} = [num2str(goodmean,'%.3f') ' (' num2str(goodstd,'%.3f') ')'];
%         %input.data(dbstrat,4) = goodstd;
%         
%         
%         % MAGMAN
%         dbs = experimentcellM{dbstrat}; 
%         experiment = dbs{2};
%         rewtrajs = load_experiment_results(experiment);
%         if (size(rewtrajs,1) < numtrials) 
%             extramsg = [' Based on at least ' num2str(size(rewtrajs,1)) ' experiments'];
%         end
%         rewtrajs = rewtrajs(1:min(numtrials,size(rewtrajs,1)),end-999:end);
%         
%         sum         = length(rewtrajs(:));
%         goodones    = rewtrajs(rewtrajs > threshold);
%         goodp       = 100 * (length(goodones)/sum);
%         goodmean    = mean(goodones);
%         goodstd     = std(goodones);
%         
%         input.data{dbstrat,4} = num2str(mean(rewtrajs(:)) , '%.2f' );
%         input.data{dbstrat,5} = num2str(goodp, '%.1f \\%%' );
%         input.data{dbstrat,6} = [num2str(goodmean,'%.3f') ' (' num2str(goodstd,'%.3f') ')'];
%     
%         
%   
%     end
%     
%     
%     
%     
%     
%     
    for dbstrat = 1:length(experimentcellS)
        dbs = experimentcellS{dbstrat}; 
        name = dbs{1};     
    
        input.tableRowLabels{dbstrat} = name;
        for c = 1:6
            if best(c,dbstrat)==0.5 
                prefix = '\textbf{';
                postfix = '}';
            elseif best(c,dbstrat)==1 
                prefix = '\textbf{\textit{';
                postfix = '}}';
            else
                prefix = '';
                postfix = '';
            end
            if c==1 || c==4
               format = '%.2f';
            elseif c==2 || c==5
                format = '%.1f \\%%';
            elseif c==3 || c==6
                format = '%.3f';
            end
            input.data{dbstrat,c} = [prefix num2str(mean(squeeze(values(c,dbstrat,:))),format) postfix];    
        end
    end
    

% % Switch transposing/pivoting your table:
% input.transposeTable = 0;
%
% % Determine whether input.dataFormat is applied column or row based:
% input.dataFormatMode = 'column'; % use 'column' or 'row'. if not set 'column' is used
%
% % Formatting-string to set the precision of the table values:
% % For using different formats in different rows use a cell array like
% % {myFormatString1,numberOfValues1,myFormatString2,numberOfValues2, ... }
% % where myFormatString_ are formatting-strings and numberOfValues_ are the
% % number of table columns or rows that the preceding formatting-string applies.
% % Please make sure the sum of numberOfValues_ matches the number of columns or
% % rows in input.tableData!
% %
input.dataFormat = {'%.3f'}; % uses three digit precision floating point for all data values
% input.dataFormat = {'%.2f',6,'%.1f\\%%',1,'%.3f',1}; % three digits precision for first two columns, one digit for the last
%
% % Define how NaN values in input.tableData should be printed in the LaTex table:
input.dataNanString = '-';
%
% % Column alignment in Latex table ('l'=left-justified, 'c'=centered,'r'=right-justified):
 input.tableColumnAlignment = 'c';
%
% % Switch table borders on/off:
 input.tableBorders = 1;
%
% % Switch table booktabs on/off:
 input.booktabs = 1;
%
% % LaTex table caption:
 input.tableCaption = [tablename extramsg];
%
% % LaTex table label:
input.tableLabel = tablelabel;

% % Switch to generate a complete LaTex document or just a table:
% input.makeCompleteLatexDocument = 1;
%
% % % Now call the function to generate LaTex code:
 latex = latexTable(input);
      
 latex2 = {};
 
 for i=1:4
     latex2{i} = latex{i};
 end
 latex2{3} = '\begin{tabular}{lccc|ccc}'; 
 latex2{5} = '& \multicolumn{3}{c}{Swingup} & \multicolumn{3}{c}{Magman} \\';
 latex2{6} = [column1str ' & mean & \Good & \subopt & mean & \Good & \subopt \\'];
 for i=7:7+length(experimentcellS) + 5
     latex2{i} = latex{i-1};
 end
 clc;
 for i=1:length(latex2)
     disp(latex2{i})
 end
end

