function [  ] = plot_experiments_rewhistogram( experimentcell, figtitle, numtrials,binwidth )

    fig = figure();
    title(figtitle)
    colorOrderT = get(gca, 'ColorOrder');
    hold on;

    for dbstrat = 1:length(experimentcell)
        dbs = experimentcell{dbstrat}; 
        dbsname = dbs{1};
        experiment = dbs{2};
        rewtrajs = load_experiment_results(experiment);
        rewtrajs = rewtrajs(1:numtrials,end-999:end);
        histogram(rewtrajs(:),'Normalization','probability','BinWidth',binwidth);
        %names{dbstrat} = [dbsname ' (' num2str(size(rewtrajs,1)) ')']; 'Color',colorOrderT(dbstrat,:)
        names{dbstrat} = dbsname;
    end
    
    
    legend(names,'location','SouthWest')
    xlabel('Reward')
    ylabel('Probability')
                
            
    

end

