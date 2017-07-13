clear; clc

experiment_data = cell(4,20);
seq_idcs = cell(1,20);
for i=1:20
    experiment_data{1,i} =  load(['./DB_10k_fifo_priois/test' num2str(i*100) '.mat']);
    experiment_data{1,i}.name = ['FIFO uniform ' num2str(i*100)];
    
    experiment_data{2,i} =  load(['./DB_10k_exp5_uni/test' num2str(i*100) '.mat']);
    experiment_data{2,i}.name = ['exp uniform ' num2str(i*100)];
    
    experiment_data{3,i} =  load(['./DB_10k_exp5_priois/test' num2str(i*100) '.mat']);
    experiment_data{3,i}.name = ['exp prioritized ' num2str(i*100)];
end
clear i


%%
figure; hold on;
filename = 'TDE_per_ep.gif';
for j = 1:20
    clf;
    hold on;
    for i = 1:3
        tempTDE = (experiment_data{i,j}.TDE);
        tempTDE(not(isfinite(tempTDE))) = NaN;
        subplot(1,3,i)
        scatter(experiment_data{i,j}.sequence_indices,tempTDE)
        axis([0 2000 0 0.1]);
    end
    legend('FIFO uniform','exp uniform','exp prioritized' )
    xlabel('experiences from sequence');ylabel('average TDE')
    title(['State after episode ' num2str(j*100)]);
    
    drawnow;
    % gif
    frame = getframe();
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if j == 1;
%       imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
%       imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end
    
%% TDE SxA space (first n experiences)
a = 1;
n = 10000;

for i = 1:3
    name = experiment_data{i,1}.name;
    figure; hold on;
    filename = ['TDE_SA_' name '.gif'];

    for j = 1:20
        clf;
        hold on;
        scatter3(experiment_data{i, j}.state(1,a:n),experiment_data{i, j}.state(2,a:n),experiment_data{i, j}.action(1,a:n),[],experiment_data{i, j}.TDE(a:n,1))
        view(45,45)
        grid;
        xlabel('position');ylabel('velocity');zlabel('action');
        title(['TDE after episode ' num2str(j*100)]);
        axis([-1 1 -1 1 -1 1])
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


for i = 1:3
    name = experiment_data{i,1}.name;
    figure; hold on;
    filename = ['TDE_S_' name '.gif'];

    for j = 1:20
        n = 10000;
        clf;
        hold on;
       %scatter(experiment_data{i, j}.state(1,a:n),experiment_data{i, j}.state(2,a:n),[],experiment_data{i, j}.TDE(a:n,1))
        scatter(experiment_data{i, j}.state(1,a:n),experiment_data{i, j}.state(2,a:n),[],1./((1+j*100)-(experiment_data{i, j}.USECOUNT(a:n,1)))); 
       
       
        xlabel('position');ylabel('velocity')
        title(['TDE of all experiences after episode ' num2str(j*100)]);
        axis([-1 1 -1 1])
        drawnow;
        % gif
        frame = getframe();
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if j == 1;
%           imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        else
%           imwrite(imind,cm,filename,'gif','WriteMode','append');
        end
    end
end




