function line = addToCostPlot( rewardtrajectories, color, fighan, conf )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

figure(fighan);
hold on
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
%line = plot(1:size(rewardtrajectories,2),mean(rewardtrajectories,1),'Color',color,'LineWidth',2);
hold off



end

