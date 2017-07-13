function [ rewardtrajectories ] = load_experiment_results( experiment, cap )
    if nargin == 1 
        cap = -1;
    end
    nrResults = 0;
    filename = ['../data/' num2str(experiment) '/RESULT_' sprintf('%03d.mat',nrResults+1)];
%     disp(filename)
    while exist(filename, 'file')==2 
        try 
            load(filename);
            nrtrials = size(seq,2);
            episodelength = size(seq,1);
        catch
            disp('break')
            break;
        end
        nrResults  = nrResults + 1;
        filename = ['../data/' num2str(experiment) '/RESULT_' sprintf('%03d.mat',nrResults+1)];
    end
    %disp(['Experiment ' num2str(experiment) ', found ' num2str(nrResults) ' results (requested: ' num2str(cap) ')' ])
    if cap == -1
        cap = nrResults;
    end
        
    if nrResults < cap
        disp('Less experiments than cap, returning nil')
        return
    end
    try
        rewardtrajectories = nan(cap,episodelength);
    catch 
        disp(experiment)
    end
%     postrajectories = zeros(episodelength,nrResults);
    
     for i =1:cap
        filename = ['../data/' num2str(experiment) '/RESULT_' sprintf('%03d.mat',i)];
        load(filename)
        rewardtrajectories(i,:) = seq;
    end
end

