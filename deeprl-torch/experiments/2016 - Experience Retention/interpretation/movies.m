%% reset the workspace
clear all;clc;close all;
load('RESULT_002', 'KernelDensity','sample_loc','sample_age');

SAVEMOVIE = true;

FIGURE_WIDTH = 700;
FIGURE_HEIGTH = 500;


timesteps = size(sample_age,2);
CUTOFF = 200;
timeres = size(KernelDensity,4)/timesteps;
velres = size(KernelDensity,1);
posres = size(KernelDensity,2);
        

%% 
plotnumber = 1; % 1 = FIFO, 2 = TDE, 3 = All, 4 = DIST per EP


ffwds               = 10;
fps                 = 30;
ffwt1               = 1:fps*ffwds;
totalsamplesffw     = 800;

frames1 = 4*fps;
frames2 = 3.5*fps;
frames3 = 3.5*fps;
frames4 = 5*fps;
frames5 = 3.5*fps;
frames6 = 12.5*fps;
frames7 = 8*fps;

swoosh_sine = sin(0:pi/(frames6-1):pi);
sinesum = sum(swoosh_sine);
sinemul = (820-15)/sinesum;
swoosh = 15+cumsum(sinemul*swoosh_sine);



switch plotnumber
    case 1
        sample_data_plot = 1;
        times = [1:10/(frames1-1):10 10*ones(1,frames2) 10:5/(frames3-1):15 15*ones(1,frames4+frames5) swoosh 820*ones(1,frames7)]; 
    case 2
        sample_data_plot = 2;
    case 4
        sample_data_plot = 3;
        times = [1:10/(frames1-1):10 10*ones(1,frames2+frames3+frames4) 10:5/(frames5-1):15 swoosh 820*ones(1,frames7)]; 
    otherwise
        sample_data_plot = -1;
        disp('No sample data')
end


   if SAVEMOVIE
    v = VideoWriter(['fifo.avi']);
    v.FrameRate = 30;
    open(v)
   end
    fig = figure('Position',[1 1 FIGURE_WIDTH FIGURE_HEIGTH ]);
    
    %times = [1:10/150:10 10:3/150/13 13:2/200:15 16:3:2000] ;
    %times = [800:2:820];
    
    for t = times
        timestep = floor(t);
        densts = max(1,floor(timeres*timestep));
        
        
        height = squeeze(KernelDensity(:,:,plotnumber,densts))./sum(sum(KernelDensity(:,:,plotnumber,densts)));
        [vel,pos] = meshgrid(-0.4:0.8/(velres-1):0.4,0:(0.1/(posres-1)):0.1);
        
        samplesx = squeeze(sample_loc(1,:,timestep,sample_data_plot));
        samplesy = squeeze(sample_loc(2,:,timestep,sample_data_plot));
        samplesc = age_color(squeeze(sample_age(:,timestep,sample_data_plot))-mod(t,1));
        disp(samplesc(1))
        
        pos = pos.';
        vel = vel.';

        % 2d plot
        p = pcolor( pos, vel, height);hold on;
         colormap('summer');
        set(p,'LineStyle','none')
        scatter(samplesx,samplesy,[],samplesc,'filled');
        %set(gca,'Color',[0.0784313753247261 0.168627455830574 0.549019634723663])
        hold off;
        
        
        xlabel('Position [m]');ylabel('Velocity [m/s]');
        title(['FIFO - Episode ' num2str(timestep)])
        axis([0 0.1 -0.4 0.4] )
        % 3d plot
        %mesh( pos, vel, height)
        
        %cleanfigure('minimumPointsDistance',1000)
        %zlabel('Relative sample density')
%       
%         axis([0 0.1 -0.4 0.4 -0.1 scaleZ(timestep)] )
%         axis([0 0.1 -0.4 0.4 0 0.01] )
%          view(timestep,45+20*sin(timestep/10));
        hold on;
%         scatter3(scatterpos, scattervel, scatterz, 'r');
        drawnow
        hold off;
        if SAVEMOVIE 
            frame = getframe(fig);    
            writeVideo(v,frame);
        end
    end
    if SAVEMOVIE
        close(v)        
    end
        
