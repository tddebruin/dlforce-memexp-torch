%experimentsToCompare = [44];
experimentsToCompare = 3:7;

explength = 5000;
means = zeros(explength,length(experimentsToCompare));
meansG = zeros(explength,length(experimentsToCompare));
for i = experimentsToCompare
    expnr = i;

    %%
    nrResults = 0;
    bestR = -100; bestER = 0;
    bestRHF = -100; bestERHF = 0;

    filename = ['../data/' num2str(expnr) '/RESULT_' sprintf('%03d.mat',nrResults+1)];
    while exist(filename, 'file')==2 
        try 
            load(filename);
            nrtrials = size(seq,1);
            %episodelength = size(BestPolStateTraj,2);
        catch
            disp('break')
            break;
        end
        nrResults  = nrResults + 1;
        filename = ['../data/' num2str(expnr) '/RESULT_' sprintf('%03d.mat',nrResults+1)];
    end
    disp(['Found ' num2str(nrResults) ' results'])
    rewardtrajectories = zeros(nrtrials,nrResults);
    %rewardtrajectoriesG = zeros(nrtrials,nrResults);
    %postrajectories = zeros(episodelength,nrResults);
    for i =1:nrResults
        filename = ['../data/' num2str(expnr) '/RESULT_' sprintf('%03d.mat',i)];
        load(filename)
%         if BestPolReward > bestR 
%             bestR = BestPolReward;
%             bestER = i;
%         end
%         if BestPolRewardHF > bestRHF
%            bestRHF = BestPolRewardHF;
%            bestERHF = i;
%         end 
        rewardtrajectories(:,i) = seq;
        %rewardtrajectoriesG(:,i) = polrewardsGIP;
        %postrajectories(:,i) = BestPolStateTraj(1,:);
    end


    means(:,expnr) = mean(rewardtrajectories,2);
    %meansG(:,expnr) = mean(rewardtrajectoriesG,2);
    %%
%     color = [1 0 0];
%     SEM = std(rewardtrajectories.')/sqrt(size(rewardtrajectories,2));               % Standard Error
%     ts = tinv([0.05  0.95],size(rewardtrajectories,2)-1);      % T-Score
%     CI = [mean(rewardtrajectories,2) mean(rewardtrajectories,2)] + (ts.'*SEM).';                      % Confidence Intervals
%     figure; hold on;
%     plot(mean(rewardtrajectories,2),'color',color,'LineWidth',2);
%     plot(CI,'color',color)
%     plot(mean(rewardtrajectoriesG,2),'b')
%     title(num2str(expnr))
end

%%
figure;
plot(means(:,3:7))
legend('SMALL DB, FIFO','FULL DB','SPLIT SMALL DB, 10% OFFPOL','SPLIT SMALL DB, 25% OFFPOL','SPLIT SMALL DB, 49% OFFPOL','Location','SouthEast')
title('Performance on training task')



%%
h = 500;
w = 350;
close all
figure('Position',[1 1 h w])
colorOrder = get(gca, 'ColorOrder');
plot(means(:,[44 46 48]))

legend('\alpha = 0','\alpha = 0.50','\alpha = 1','Location','SouthEast')
%title('Performance on training task')
xlabel('Episode');ylabel('Average reward');


figure('Position',[h+1 1 h w])
plot(meansG(:,[44 46 48]))
legend('\alpha = 0','\alpha = 0.50','\alpha = 1','Location','SouthEast')
%title('Performance on generalization task')
xlabel('Episode');ylabel('Average reward');


%%
h = 500;
w = 350;
close all
figure('Position',[1 1 h w])
plot(means(:,[44 49 50 52]))
legend('FIFO','Hybrid \beta = 0.25','Hybrid \beta = 0.5','TDE','Location','SouthEast')
%title('Performance on training task')
xlabel('Episode');
ylabel('Average reward');
%set(h,'interpret','LaTeX')

figure('Position',[h+1 1 h w])
plot(meansG(:,[44 49 50 52]))
legend('FIFO','Hybrid \beta = 0.25','Hybrid \beta = 0.5','TDE','Location','SouthEast')
%title('Performance on generalization task')
xlabel('Episode');ylabel('Average reward');


%%
h = 500;
w = 350;
%close all
figure('Position',[1 1 h w])
plot(means(:,[44 53 54 55 56]))
legend('0% uniform actions','25% uniform actions','50% uniform actions','75% uniform actions','100% uniform actions','Location','SouthEast')
title('Performance on training task')
xlabel('Episode');
ylabel('Average reward');
%set(h,'interpret','LaTeX')

figure('Position',[h+1 1 h w])
plot(meansG(:,[44 53 54 55 56]))
legend('0% uniform actions','25% uniform actions','50% uniform actions','75% uniform actions','100% uniform actions','Location','SouthEast')
title('Performance on generalization task')
xlabel('Episode');ylabel('Average reward');

