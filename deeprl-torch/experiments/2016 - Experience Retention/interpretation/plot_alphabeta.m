clear all;


CONFIDENCE = 0.9;

plot1experiments = [1 2]; names1 = {'DB samples','0.8 FA'}.';
EXPERIMENT1_CAP = 9; 


% figure sizes:
h = 500;
w = 350;
LegendLocation = 'SouthEast';

%% plot 1: effect of hypothetical uniform experience fraction
close all
clc;

figT = figure('Position',[1 1 h w], 'Name','EndGame Trials','NumberTitle','off');
colorOrderT = get(gca, 'ColorOrder');
counter = 1;
for i=plot1experiments
    [rew] = load_experiment_results(i,EXPERIMENT1_CAP);
    addToCostPlot( rew, colorOrderT(counter,:), figT, CONFIDENCE );
    counter = counter + 1;
end
counter = 1;
for i=plot1experiments
    [rew] = load_experiment_results(i,EXPERIMENT1_CAP);
    figure(figT); hold on;
    legend_t(i) = plot(1:size(rew,2),mean(rew,1),'Color',colorOrderT(i,:),'LineWidth',1);
end


figure(figT);
l = legend(legend_t,names1);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');

%%

%% plot 2: our method performance

figT = figure('Position',[1 150+w h w], 'Name','Method performance: task','NumberTitle','off');
colorOrderT = get(gca, 'ColorOrder');
figG = figure('Position',[100+h 150+w h w], 'Name','Method performance: generalization','NumberTitle','off');
counter = 1;
for i=plot2experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT2_CAP);
    addToCostPlot( rew, colorOrderT(counter,:), figT, CONFIDENCE );
    addToCostPlot( rewG, colorOrderT(counter,:), figG, CONFIDENCE );
    counter = counter + 1;
end

counter = 1;
for i=plot2experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT2_CAP);
    figure(figT); hold on;
    legend_t(counter) = plot(1:size(rew,2),mean(rew,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    figure(figG); hold on;
    legend_g(counter) = plot(1:size(rewG,2),mean(rewG,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    counter = counter + 1;
end

figure(figT);
% cleanfigure('minimumPointsDistance',100)

%legend('\alpha = 0','\alpha = 0.50','\alpha =  1','Location','SouthEast')
l = legend(legend_t,names2);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');
figure(figG);
l = legend(legend_g,names2);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');

%% plot 2B: our method performance - movie

plot2experiments = [44 49 ]; names2 = {'STANDARD FIFO','OUR METHOD'}.';
%plot2experiments = [44 49 52]; names2 = {'FIFO','Hybrid, \beta = 0.25','TDE'}.';

EXPERIMENT2_CAP = 30;

% figure sizes:
h = 400;
w = 400;



figT = figure('Position',[1 150+w h w], 'Name','Method performance: task','NumberTitle','off');
colorOrderT = get(gca, 'ColorOrder');
figG = figure('Position',[100+h 150+w h w], 'Name','Method performance: generalization','NumberTitle','off');
counter = 1;
for i=plot2experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT2_CAP);
    addToCostPlot( rew, colorOrderT(counter,:), figT, CONFIDENCE );
    addToCostPlot( rewG, colorOrderT(counter,:), figG, CONFIDENCE );
    counter = counter + 1;
end

counter = 1;
for i=plot2experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT2_CAP);
    figure(figT); hold on;
    legend_t(counter) = plot(1:size(rew,2),mean(rew,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    figure(figG); hold on;
    legend_g(counter) = plot(1:size(rewG,2),mean(rewG,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    counter = counter + 1;
end

figure(figT);
% cleanfigure('minimumPointsDistance',100)

%legend('\alpha = 0','\alpha = 0.50','\alpha =  1','Location','SouthEast')
%l = legend(legend_t,names2);
%set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');
figure(figG);
%l = legend(legend_g,names2);
%set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');


%%
%% plot 3: effect of hypothetical uniform action fraction
close all
clc;

figT = figure('Position',[1 1 h w], 'Name','Uniform experiences: task','NumberTitle','off');
colorOrderT = get(gca, 'ColorOrder');
figG = figure('Position',[100+h 1 h w], 'Name','Uniform experiences: generalization','NumberTitle','off');
counter = 1;
for i=plot3experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT3_CAP);
    addToCostPlot( rew, colorOrderT(counter,:), figT, CONFIDENCE );
    addToCostPlot( rewG, colorOrderT(counter,:), figG, CONFIDENCE );
    counter = counter + 1;
end
counter = 1;
for i=plot3experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT3_CAP);
    figure(figT); hold on;
    legend_t(counter) = plot(1:size(rew,2),mean(rew,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    figure(figG); hold on;
    legend_g(counter) = plot(1:size(rewG,2),mean(rewG,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    counter = counter + 1;
end


figure(figT);
%legend('\alpha = 0','\alpha = 0.50','\alpha =  1','Location','SouthEast')
l = legend(legend_t,names3);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');
figure(figG);
l = legend(legend_g,names3);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');


%% plot 4: effect of hypothetical uniform action fraction vs uniform state action
close all
clc;

figT = figure('Position',[1 1 h w], 'Name','Uniform experiences: task','NumberTitle','off');
colorOrderT = get(gca, 'ColorOrder');
figG = figure('Position',[100+h 1 h w], 'Name','Uniform experiences: generalization','NumberTitle','off');
counter = 1;
for i=plot4experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT4_CAP);
    addToCostPlot( rew, colorOrderT(counter,:), figT, CONFIDENCE );
    addToCostPlot( rewG, colorOrderT(counter,:), figG, CONFIDENCE );
    counter = counter + 1;
end
counter = 1;
for i=plot4experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT4_CAP);
    figure(figT); hold on;
    legend_t(counter) = plot(1:size(rew,2),mean(rew,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    figure(figG); hold on;
    legend_g(counter) = plot(1:size(rewG,2),mean(rewG,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    counter = counter + 1;
end


figure(figT);
%legend('\alpha = 0','\alpha = 0.50','\alpha =  1','Location','SouthEast')
l = legend(legend_t,names4);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');
figure(figG);
l = legend(legend_g,names4);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');


%% plot 5: LONGRUN
close all
clc;

figT = figure('Position',[1 1 h w], 'Name','Uniform experiences: task','NumberTitle','off');
colorOrderT = get(gca, 'ColorOrder');
figG = figure('Position',[100+h 1 h w], 'Name','Uniform experiences: generalization','NumberTitle','off');
counter = 1;
for i=plot5experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT5_CAP);
    addToCostPlot( rew, colorOrderT(counter,:), figT, CONFIDENCE );
    addToCostPlot( rewG, colorOrderT(counter,:), figG, CONFIDENCE );
    counter = counter + 1;
end
counter = 1;
for i=plot5experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT5_CAP);
    figure(figT); hold on;
    legend_t(counter) = plot(1:size(rew,2),mean(rew,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    figure(figG); hold on;
    legend_g(counter) = plot(1:size(rewG,2),mean(rewG,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    counter = counter + 1;
end


figure(figT);
%legend('\alpha = 0','\alpha = 0.50','\alpha =  1','Location','SouthEast')
l = legend(legend_t,names4);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');
figure(figG);
l = legend(legend_g,names4);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');


%% plot 6: LONGRUN - selective 
close all
clc;

figT = figure('Position',[1 1 h w], 'Name','Uniform experiences: task','NumberTitle','off');
colorOrderT = get(gca, 'ColorOrder');
figG = figure('Position',[100+h 1 h w], 'Name','Uniform experiences: generalization','NumberTitle','off');
counter = 1;
for i=plot6experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT5_CAP);
    addToCostPlot( rew, colorOrderT(counter,:), figT, CONFIDENCE );
    addToCostPlot( rewG, colorOrderT(counter,:), figG, CONFIDENCE );
    counter = counter + 1;
end
counter = 1;
for i=plot6experiments
    [rew, rewG] = load_experiment_results(i,EXPERIMENT5_CAP);
    figure(figT); hold on;
    legend_t(counter) = plot(1:size(rew,2),mean(rew,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    figure(figG); hold on;
    legend_g(counter) = plot(1:size(rewG,2),mean(rewG,1),'Color',colorOrderT(counter,:),'LineWidth',1);
    counter = counter + 1;
end


figure(figT);
%legend('\alpha = 0','\alpha = 0.50','\alpha =  1','Location','SouthEast')
l = legend(legend_t,names4);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');
figure(figG);
l = legend(legend_g,names4);
set(l,'Location',LegendLocation);
xlabel('Episode');ylabel('Reward');


