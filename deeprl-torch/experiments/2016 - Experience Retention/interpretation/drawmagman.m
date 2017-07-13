%% Drawing of magman

c1 = 5.52e-10;
c2 = 1.75e-4;

x = -0.035:1e-3:0.105;

magnetforces = zeros(4,length(x));
figure(1)
hold on;
for m=1:4
    for i=1:length(x)
        magnetforces(m,i) = (-c1*(x(i)-0.025*m))/((x(i)-0.025*m)^2+c2)^3;
    end
    %linecolor = [0.7 0.7 0.7];
    linecolor = [1 1 1];
    linewidth = 1;
    linestyle = '-';
    %if m==3
    %    linecolor = [0 0 0];
    %    linewidth = 1;
    %    linestyle = '--';
    %end
        %plot(x,magnetforces(m,:),'color',linecolor,'LineWidth',linewidth,'LineStyle',linestyle)
        plot(x,magnetforces(m,:),'LineWidth',linewidth)
end


