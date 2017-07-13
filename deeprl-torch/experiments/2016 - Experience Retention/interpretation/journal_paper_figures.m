%% journal trajectories


%% swingup 50Hz
clear all
load('traj-swingup.mat')
ref = 1;
ts = 0.02;

%% swingup 100Hz
clear all
load('traj-swingup100.mat')
ref = 1;
ts = 0.01;

%% magman 50Hz
clear all
load('traj-magman.mat')
ref = 0;
ts = 0.02;

%%


state = permute(state, [3 2 1]);
action = permute(action, [3 2 1]);
reward = (permute(reward, [2 1]))*10;
seq_req = mean(reward,2);

%%

plot(seq_req)
%plot(abs(seq_req+0.4))

%% best:
idx = find(seq_req == max(seq_req));
figure(1);
time = ts:ts:4;
dref = ref - abs(squeeze(state(idx,:,1)));
plot(time,dref)
figure(2)
plot(time,squeeze(reward(idx,:)))
figure(3)
plot(squeeze(action(idx,:,:)))

%% 0.4
idx = find((abs(seq_req+0.4)) == min(abs(seq_req+0.4)));
seq_req(idx);
figure(1);
time = 0.02:0.02:4;
dref = ref - abs(squeeze(state(idx,:,1)));
plot(time,dref)



%% Q
gamma = 0.1;
Q = zeros(1,size(reward,2));
for i = (length(Q)-1):-1:1
    Q(i) = reward(idx,i) + gamma*Q(i+1);
end
plot(time,reward(idx,:),time,Q)

