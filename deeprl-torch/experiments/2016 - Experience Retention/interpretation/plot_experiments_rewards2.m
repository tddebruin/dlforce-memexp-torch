function [  ] = plot_experiments_rewards2( experimentcell, figtitle )

    fig = figure();
    title(figtitle)
    colorOrderT = get(gca, 'ColorOrder');
    hold on;

    for dbstrat = 1:length(experimentcell)
        dbs = experimentcell{dbstrat}; 
        dbsname = dbs{1};
        experiment = dbs{2};
        rewtrajs = load_experiment_results(experiment);
        plot(1:size(rewtrajs,2),mean(rewtrajs,1),'Color',colorOrderT(dbstrat,:));
        %names{dbstrat} = [dbsname ' (' num2str(size(rewtrajs,1)) ')'];
        names{dbstrat} = dbsname;
    end
    
    
    legend(names,'location','SouthWest')
    xlabel('Episode')
    ylabel('Average reward')
                
            
    

end

