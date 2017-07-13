clear all;
clc;
close all;

load('diff_db_lrhf.mat', 'KernelDensity')
%% 1
for i=1:3
    h = 500;
    w = 350;
    figure('Position',[1 1 h w])
    height = squeeze(KernelDensity(:,:,i,200))./sum(sum(KernelDensity(:,:,1,200)));
    [vel,pos] = meshgrid(-0.4:0.8/79:0.4,0:(0.1/99):0.1);

    pos = pos.';
    vel = vel.';

    mesh( pos, vel, height)
%cleanfigure('minimumPointsDistance',1000)
    xlabel('Position [m]');ylabel('Velocity [m/s]');zlabel('Relative sample density')
end

%% 2
plotnumber = 2; % 1 = FIFO, 2 = TDE, 3 = DIST per samp, 4 = DIST per EP
timesteps = 1000;
velres = 40;
posres = 50;
% startH = 0.05;
% endH = 0.0065;
startH = 0.08;
endH = 0.08;
zheight = zeros(1,timesteps);
scaleZ = zeros(1,timesteps);
for i=1:timesteps    
    zheight(i) = max(max(squeeze(KernelDensity(:,:,plotnumber,i))./sum(sum(KernelDensity(:,:,plotnumber,i)))));
    scaleZ(i) = endH + (startH-endH)*exp(-((i-1)*0.01));
end   

plot(zheight);hold on; plot(scaleZ)
%%
    %v = VideoWriter(['test.mj2'],'Motion JPEG 2000');
    %v.CompressionRatio = 2;
    v = VideoWriter(['test.avi']);
    v.FrameRate = 25;
    open(v)

    
    
    velres = 40;
    posres = 50;
    
    h = 1000;
    w = 700;
    fig = figure('Position',[1 1 h w]);
    
    scattervel = [0 0.00 0.001]; 
    scatterpos = [0 0.035 0.001]; 
    scatterz = [0 0 -0.1];
    
    for timestep = 1:1000

        height = squeeze(KernelDensity(:,:,plotnumber,timestep))./sum(sum(KernelDensity(:,:,plotnumber,timestep)));
        [vel,pos] = meshgrid(-0.4:0.8/(velres-1):0.4,0:(0.1/(posres-1)):0.1);

        pos = pos.';
        vel = vel.';

        mesh( pos, vel, height)
        
        %cleanfigure('minimumPointsDistance',1000)
        xlabel('Position [m]');ylabel('Velocity [m/s]');zlabel('Relative sample density')
        axis([0 0.1 -0.4 0.4 -0.1 scaleZ(timestep)] )
        %axis([0 0.1 -0.4 0.4 0 0.08] )
         view(timestep,45+20*sin(timestep/10));
        hold on;
        scatter3(scatterpos, scattervel, scatterz, 'r');
        drawnow
        frame = getframe(fig);    
        hold off;
        writeVideo(v,frame);
    end
    close(v)        


%% video



        fig = figure('units','pixels','Position',[0 0 360 720]);
        time = (1:size(Angles,1))/100;
        totime = max(time);

        k = 1;
        for timeindex = 0:(1/FPS):totime
            subplot(2,1,1,'replace');
            plotnumber(time,ResponseX_3,'LineWidth',2,'Color','b'); hold on;
            plotnumber(time,ResponseX_6,'LineWidth',2,'Color','r'); hold on;
            plotnumber(time,refx,'LineWidth',2,'Color','k');
            axis([(timeindex-xscale) timeindex  -1.8 1.8]);    
            xlabel('Time [s]');ylabel('Xpos ');
            subplot(2,1,2,'replace');
            plotnumber(time,ResponseY_3,'LineWidth',2,'Color','b'); hold on;
            plotnumber(time,ResponseY_6,'LineWidth',2,'Color','r'); hold on;
            plotnumber(time,refy,'LineWidth',2,'Color','k');
            axis([(timeindex-xscale) timeindex  0 3]);
            xlabel('Time [s]');ylabel('Ypos ');
            drawnow
            frame = getframe(fig);    

            writeVideo(v,frame);
        end
        close(v)
        
