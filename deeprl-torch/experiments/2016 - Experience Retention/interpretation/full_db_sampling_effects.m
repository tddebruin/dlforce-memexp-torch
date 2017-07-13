clear; clc

experiment_data = cell(4,20);
seq_idcs = cell(1,20);
for i=1:20
    experiment_data{1,i} =  load(['./DB_full_uniform/test' num2str(i*100) '.mat']);
    experiment_data{1,i}.name = ['uniform after episode ' num2str(i*100)];
    
    experiment_data{2,i} =  load(['./DB_full_prioritized/test' num2str(i*100) '.mat']);
    experiment_data{2,i}.name = ['naive prioritized after episode ' num2str(i*100)];
    
    experiment_data{3,i} =  load(['./DB_full_FSprioritized/test' num2str(i*100) '.mat']);
    experiment_data{3,i}.name = ['DM-Rank prioritized after episode ' num2str(i*100)];
    
    experiment_data{4,i} =  load(['./DB_full_FSprioritized_IS/test' num2str(i*100) '.mat']);
    experiment_data{4,i}.name = ['DM-Rank prioritized with importance sampling after episode ' num2str(i*100)];

    seq_idcs{i} = reshape(experiment_data{1,i}.sequence_indices(1:i*100*199),199,[]);
end
clear i


%%
figure; hold on;
filename = 'TDE_per_ep.gif';
for j = 1:20
    clf;
    hold on;
    for i = 1:4 
        tempTDE =  reshape(experiment_data{i,j}.TDE(1:j*100*199),199,[]);
        tempTDE(not(isfinite(tempTDE))) = NaN;
        plot(mean(seq_idcs{j}),mean(tempTDE,'omitnan'))
    end
    legend('uniform','naive prioritized','DM rank-based','RM rank-based with IS')
    xlabel('experiences from sequence');ylabel('average TDE')
    title(['State after episode ' num2str(j*100)]);
    axis([0 2000 0 0.05]);
    drawnow;
    % gif
    frame = getframe();
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if j == 1;
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
      imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end
    
%% TDE SxA space (first n experiences)
a = 1;
n = 10000;

for i = 1:4
    name = experiment_data{i,1}.name;
    figure; hold on;
    filename = ['TDE_SA_' name '.gif'];

    for j = 1:20
        clf;
        hold on;
        scatter3(experiment_data{i, j}.state(1,a:n),experiment_data{i, j}.state(2,a:n),experiment_data{i, j}.action(1,a:n),[],experiment_data{i, j}.TDE(a:n,1))
        view(45,45)
        xlabel('position');ylabel('velocity');zlabel('action');
        title(['TDE of first ' num2str(n) ' experiences after episode ' num2str(j*100)]);
        
        drawnow;
        % gif
        frame = getframe();
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if j == 1;
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        else
          imwrite(imind,cm,filename,'gif','WriteMode','append');
        end
    end
end


%% TDE S space (all experiences)
a = 1;


for i = 4:4
    name = experiment_data{i,1}.name;
    figure; hold on;
    filename = ['TDE_S_' name '.gif'];

    for j = 1:20
        n = j*100*199;
        clf;
        hold on;
        scatter(experiment_data{i, j}.state(1,a:n),experiment_data{i, j}.state(2,a:n),[],experiment_data{i, j}.TDE(a:n,1))
        
        xlabel('position');ylabel('velocity')
        title(['TDE of all experiences after episode ' num2str(j*100)]);
        
        drawnow;
        % gif
        frame = getframe();
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if j == 1;
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        else
          imwrite(imind,cm,filename,'gif','WriteMode','append');
        end
    end
end




%% REWARDS after 2000 episodes
figure; hold on;
for i = 1:4     
    plot(mean(seq_idcs{20}),sum( reshape(experiment_data{i,20}.reward(1:2000*199),199,[]),'omitnan')/20)
end
    legend('uniform','naive prioritized','DM rank-based','RM rank-based with IS')
    xlabel('sequence');ylabel('reward')
%%

boxplot(prioritized.TDE,prioritized.sequence_indices)