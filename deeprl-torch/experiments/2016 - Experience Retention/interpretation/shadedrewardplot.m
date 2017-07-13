function [ h ] = shadedrewardplot( experiments, cap, conf, figtitle, optimum )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    

h = figure('Position',0.9*[0 0 400 320]);
hold on
colors = get(gca, 'ColorOrder');
for experiment = 1:length(experiments)
    exp = experiments{experiment};
    
    color = colors(experiment,:);
    legendnames{experiment} = exp{1};
    rewardtrajectories = load_experiment_results(exp{2},cap);
    if(size(rewardtrajectories,2) > 1)
        SEM = std(rewardtrajectories)/sqrt(size(rewardtrajectories,1));               % Standard Error
        ts = tinv([(1-conf)  conf],size(rewardtrajectories,1)-1);      % T-Score
        CI = [mean(rewardtrajectories,1); mean(rewardtrajectories,1)] + (ts.'*SEM); 

        x=1:size(rewardtrajectories,2);         %#initialize x array
        y1=CI(1,:);
        y2=CI(2,:);
        X=[x,fliplr(x)];                %#create continuous x value array for plotting
        Y=[y1,fliplr(y2)];              %#create y values for out and then back    
        a = fill(X,Y,color);          %#plot filled area
        set(a,'EdgeColor','None');
        alpha(a,0.35)
    end
    lines(experiment) = plot(1:size(rewardtrajectories,2),mean(rewardtrajectories,1),'Color',color,'LineWidth',1);
end
if optimum > -1
    lines(length(lines)+1) = plot([0 2000],[optimum optimum],'k--');
    legendnames{length(legendnames)+1} = 'Optimal controller';
end
leg = legend(lines,legendnames,'Location','SouthEast');
%set(leg,'TextSize', 12)
xlabel('Episode');
ylabel('Mean reward per epiode')
hold off

title(figtitle)
end

