function [  ] = plot_experiments_rewards( experimentcell, figtitle )

    fig = figure();
    title(figtitle)
    colorOrderT = get(gca, 'ColorOrder');
    hold on;
    linecounter = 0;
    for dbstrat = 1:length(experimentcell)
        dbs = experimentcell{dbstrat}; 
        dbsname = dbs{1};
        for samplestrat = 2:length(dbs)
            experiment = dbs{samplestrat};
            if experiment{2} > 0 %% experiment has been run
                linecounter = linecounter + 1;
                markervar = 'none';
                rewtrajs = load_experiment_results(experiment{2});
                if strcmp(experiment{1},'PER')
                    markervar = 'x';
                elseif strcmp(experiment{1},'PER+IS')
                    markervar = '^';
                elseif strcmp(experiment{1},'PER+FIS')
                    markervar = 'd';
                elseif strcmp(experiment{1},'10% Synth S')
                    markervar = '^';
                elseif strcmp(experiment{1},'10% Synth A')
                    markervar = 'v';
                elseif strcmp(experiment{1},'10% Synth SA')
                    markervar = 'd';
                end
                plot(1:size(rewtrajs,2),mean(rewtrajs,1),'Color',colorOrderT(dbstrat,:),'Marker',markervar);
                names{linecounter} = [dbsname ' ' experiment{1} ' (' num2str(size(rewtrajs,1)) ')'];
            end
        end
    end
    
    
    legend(names,'location','SouthWest')
    xlabel('Episode')
    ylabel('Average reward')
                
            
    

end

