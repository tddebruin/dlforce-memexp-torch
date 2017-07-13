clear all; clc;
figure;
rank = 1:10000;
a = 1;
pun = (1./rank).^a;
%p = pun/sum(pun);
p = cumsum(pun);

k = 800;%max(2,min(ceil((length(rank))/200),100));
indices = ceil(p/(p(end)/k));
stairs(rank,indices)

disp(['uniform probability of sampling: ' num2str(1/length(rank))])
disp(['probability of overwriting one of the 10% least desirable samples: ' num2str((1/k)*((1/(max(1,(sum(indices==1)))))))])
disp([num2str(((1/k)*((1/((max(1,sum(indices==1)))))))/(1/length(rank))) ' times uniform'])
disp(['probability of overwriting one of the 10% most desirable samples: ' num2str((1/k)*(1/(sum(indices==k))))])
disp([num2str((1/k)*(1/(sum(indices==k)))/(1/length(rank))) ' times uniform'])