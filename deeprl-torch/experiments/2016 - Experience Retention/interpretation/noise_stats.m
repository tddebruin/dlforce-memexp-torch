% 78/51: EXPL(1.0) - Uniform
% 79/51: EXPL(1.0) - PER+FIS
% 80/51 : TDE(1.0) - uniform
% 81/51:  TDE(1.0) - PER+FIS



experiment = 78
trial = 51
basedir = '../data/'
dir = [basedir num2str(experiment) '/RESULT_0' num2str(trial) '/'] 

figure(1)
clf;

% base noise
edges = -0.1:0.008:0.1;
x = [-0.1:0.001:0.1];
norm = normpdf(x,0,0.02);

%%
for i = 1:2000
    load([dir num2str(i) '.mat'])
    subplot(1,2,1);
    scatter(state1,state2)
    axis([-0.1,0.1, -.1, .1]);
    subplot(1,2,2);
    histogram(action)
    axis([-.1,.1 0 200])
    drawnow;
end
%%
for i = 1:3000
    load([dir num2str(i) '.mat'])
    subplot(1,3,1);
    h = histogram(state1,edges,'normalization','probability');
    hold on;
    plot(x,norm/125);
    axis([-.1,.1 0 0.175])
    hold off;
    subplot(1,3,2);
    
    histogram(state2,edges,'normalization','probability')
    hold on;
    plot(x,norm/125);
    axis([-.1,.1 0 0.175])
    hold off;
    subplot(1,3,3);
    
   histogram(action,edges,'normalization','probability')
   hold on;
   plot(x,norm/125);
    hold off;
    axis([-.1,.1 0 0.175])
    drawnow;
end

%%
firstIndex = 9;
lastIndex  = 3000;

dataperupdate = 1600;
updates = lastIndex - firstIndex;

s1last1000 = zeros(dataperupdate*updates,1);
s2last1000 = zeros(dataperupdate*updates,1);
alast1000 = zeros(dataperupdate*updates,1);

for i = firstIndex:lastIndex
    bi = 1+((i-firstIndex))*1600;
    li = ((i-firstIndex)+1)*1600;
    load([dir num2str(i) '.mat'])
    s1last1000(bi:li,:) = state1;
    s2last1000(bi:li,:) = state2;
    alast1000(bi:li,:) = action;
    
end
    
%% RELEVANT PLOT
    figure(1);clf;
    
    subplot(1,3,1);
    h = histogram(state1,edges,'normalization','probability');
    hold on;
    plot(x,norm/125);
    axis([-.1,.1 0 0.175])
    hold off;
    xlabel('angle')
    
    subplot(1,3,2);
    histogram(state2,edges,'normalization','probability')
    hold on;
    plot(x,norm/125);
    axis([-.1,.1 0 0.175])
    hold off;
    xlabel('angular velocity')
    
    subplot(1,3,3);
    histogram(action,edges,'normalization','probability')
    hold on;
    plot(x,norm/125);
    hold off;
    axis([-.1,.1 0 0.175])
    xlabel('action')
    
    drawnow;



%% 

histogram([s1last1000 s2last1000 alast1000],-0.1:0.008:0.1,'normalization','probability')
hold on;
plot(x,norm./(125));




%%
% 78/51: EXPL(1.0) - Uniform
% 79/51: EXPL(1.0) - PER+FIS
% 80/51 : TDE(1.0) - uniform
% 81/51:  TDE(1.0) - PER+FIS

data = { {'\method{\expl{1.0}}{\unif}',78,51},{'\method{\expl{1.0}}{\perfis}',79,51},...  
    {'\method{\tde{1.0}}{\unif}',80,51},{'\method{\tde{1.0}}{\perfis}',81,51},...
    };

latexcode = noise_table( data, 0, 0.02, 9, 3000,'Properties of the noise in the mini batches as a function of the experience selection procedure','tab::noise_stats');

%%
latexcode = noise_table2( data, 0, 0.02, 9, 3000,'Properties of the noise in the mini batches as a function of the experience selection procedure','tab::noise_stats');


