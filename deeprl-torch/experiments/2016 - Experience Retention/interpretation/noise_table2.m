function [ latex2 ] = noise_table( experimentcell, truemean, truestdv, index_start, index_end, caption, label )
    
    input.data = cell(length(experimentcell), 3);
    input.tableColLabels = {'position','velocity','action'};
    input.tableRowLabels = {'Collected experiences'};
    extramsg = '' ;
    for dbstrat = 1:length(experimentcell)
        dbs = experimentcell{dbstrat}; 
        dbsname = dbs{1};
        experiment = dbs{2};
        trial = dbs{3};
        
        
        dataperupdate = 1600;
        updates = index_end - index_start;

        s1 = zeros(dataperupdate*updates,1);
        s2 = zeros(dataperupdate*updates,1);
        a = zeros(dataperupdate*updates,1);
        
        basedir = '../data/';
        dir = [basedir num2str(experiment) '/RESULT_0' num2str(trial) '/'] ;
        
        for i = index_start:index_end
            bi = 1+((i-index_start))*1600;
            li = ((i-index_start)+1)*1600;
            load([dir num2str(i) '.mat'])
            s1(bi:li,:) = state1;
            s2(bi:li,:) = state2;
            a(bi:li,:) = action;
        end
        format = '%.3e';
        
        input.data{1,1} = num2str(truestdv*(sqrt(2/pi)) , format );
        input.data{1,2} = num2str(truestdv*(sqrt(2/pi)) , format );
        input.data{1,3} = num2str(truestdv*(sqrt(2/pi)) , format );
        
        
        input.tableRowLabels{dbstrat+1} = dbsname;
        input.data{dbstrat+1,1} = num2str(mean(abs(s1)) , format );
        input.data{dbstrat+1,2} = num2str(mean(abs(s2)) , format );
        input.data{dbstrat+1,3} = num2str(mean(abs(a)) , format );
       
        
  
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
 input.tableCaption = [caption];
%
% % LaTex table label:
input.tableLabel = label;

% % Switch to generate a complete LaTex document or just a table:
% input.makeCompleteLatexDocument = 1;
%
% % % Now call the function to generate LaTex code:
 latex = latexTable(input);
      
 latex2 = {};
 
 for i=1:5
     latex2{i} = latex{i};
 end
 latex2{3}(17) = 'l'; % first column left outline
 for i=6:length(latex)
     latex2{i} = latex{i};
 end
 clc;
 for i=1:length(latex2)
     disp(latex2{i})
 end

end

