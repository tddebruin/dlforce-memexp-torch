function [ latex2 ] = numbers_experiments_rewhistogram( experimentcellS, experimentcellM, column1str, numtrials, tablename, tablelabel, threshold )

    input.data = cell(length(experimentcellS), 6);
    input.tableColLabels = {'mean','percent good','mean, stdv of good','mean','percent good','mean, stdv of good'};
    input.tableRowLabels = {};
    extramsg = '' ;
    for dbstrat = 1:length(experimentcellS)
        dbs = experimentcellS{dbstrat}; 
        dbsname = dbs{1};
        experiment = dbs{2};
        rewtrajs = load_experiment_results(experiment);
        if (size(rewtrajs,1) < numtrials) 
            extramsg = [' Based on at least ' num2str(size(rewtrajs,1)) ' experiments'];
        end
        rewtrajs = rewtrajs(1:min(numtrials,size(rewtrajs,1)),end-999:end);
        
        sum         = length(rewtrajs(:));
        goodones    = rewtrajs(rewtrajs > threshold);
        goodp       = 100 * (length(goodones)/sum);
        goodmean    = mean(goodones);
        goodstd     = std(goodones);
        
        name        = dbsname;
    
        input.tableRowLabels{dbstrat} = name;
        input.data{dbstrat,1} = num2str(mean(rewtrajs(:)) , '%.2f' );
        input.data{dbstrat,2} = num2str(goodp, '%.1f \\%%' );
        input.data{dbstrat,3} = [num2str(goodmean,'%.3f') ' (' num2str(goodstd,'%.3f') ')'];
        %input.data(dbstrat,4) = goodstd;
        
        
        % MAGMAN
        dbs = experimentcellM{dbstrat}; 
        experiment = dbs{2};
        rewtrajs = load_experiment_results(experiment);
        if (size(rewtrajs,1) < numtrials) 
            extramsg = [' Based on at least ' num2str(size(rewtrajs,1)) ' experiments'];
        end
        rewtrajs = rewtrajs(1:min(numtrials,size(rewtrajs,1)),end-999:end);
        
        sum         = length(rewtrajs(:));
        goodones    = rewtrajs(rewtrajs > threshold);
        goodp       = 100 * (length(goodones)/sum);
        goodmean    = mean(goodones);
        goodstd     = std(goodones);
        
        input.data{dbstrat,4} = num2str(mean(rewtrajs(:)) , '%.2f' );
        input.data{dbstrat,5} = num2str(goodp, '%.1f \\%%' );
        input.data{dbstrat,6} = [num2str(goodmean,'%.3f') ' (' num2str(goodstd,'%.3f') ')'];
    
        
  
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
 latex2{3}(17) = 'l'; % first column left outline
 latex2{5} = '& \multicolumn{3}{c}{Swingup} & \multicolumn{3}{c}{Magman} \\';
 latex2{6} = [column1str ' & mean & \Good & mean, stdv of \Good & mean & \Good & mean, stdv of \Good \\'];
 for i=7:7+length(experimentcellS) + 5
     latex2{i} = latex{i-1};
 end
 clc;
 for i=1:length(latex2)
     disp(latex2{i})
 end

end

