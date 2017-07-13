
%% 
%
% Plots and numbers for the journal paper
%
%

%% Synthetic experiences

NUMEXP = 50;

%% METHODS

% 13 Magman 50Hz, 10k xp, FIFO, Uniform
% 14 Magman 50Hz, 10k xp, FIFO, PER
% 15 Magman 50Hz, 10k xp, FIFO, PER+IS
% 16 Magman 50Hz, 10k xp, EXPL a=0.25, Uniform
% 17 Magman 50Hz, 10k xp, EXPL a=0.25, PER
% 18 Magman 50Hz, 10k xp, EXPL a=0.25, PER+IS
% 19 Magman 50Hz, 10k xp, EXPL a=0.5, Uniform
% 20 Magman 50Hz, 10k xp, EXPL a=0.5, PER
% 21 Magman 50Hz, 10k xp, EXPL a=0.5, PER+IS
% 22 Magman 50Hz, 10k xp, EXPL a=1.2, Uniform
% 23 Magman 50Hz, 10k xp, EXPL a=1.2, PER
% 24 Magman 50Hz, 10k xp, EXPL a=1.2, PER+IS
% 25 Swingup 50Hz, 10k xp, FIFO, Uniform
% 26 Swingup 50Hz, 10k xp, FIFO, PER
% 27 Swingup 50Hz, 10k xp, FIFO, PER+IS
% 28 Swingup 50Hz, 10k xp, EXPL a=0.25, Uniform
% 29 Swingup 50Hz, 10k xp, EXPL a=0.25, PER
% 30 Swingup 50Hz, 10k xp, EXPL a=0.25, PER+IS
% 31 Swingup 50Hz, 10k xp, EXPL a=0.5, Uniform
% 32 Swingup 50Hz, 10k xp, EXPL a=0.5, PER
% 33 Swingup 50Hz, 10k xp, EXPL a=0.5, PER+IS
% 34 Swingup 50Hz, 10k xp, EXPL a=1.2, Uniform
% 35 Swingup 50Hz, 10k xp, EXPL a=1.2, PER
% 36 Swingup 50Hz, 10k xp, EXPL a=1.2, PER+IS

% 59 Magman 50Hz, 10k xp, EXPL a=1.2, PER+FIS
% 60 Swingup 50Hz, 10k xp, EXPL a=1.2, PER+FIS

% 63 Swingup 50Hz, 10k xp, TDE a=1.2, Uniform
% 64 Swingup 50Hz, 10k xp, TDE a=1.2, PER+FIS
% 65 Magman 50Hz, 10k xp, TDE a=1.2, Uniform
% 66 Magman 50Hz, 10k xp, TDE a=1.2, PER+FIS

%magman.standardfull = {{'fifo',{'Uniform',1},{'PER',9},{'PER+IS',10}}};
%swingup.standardfull = {{'fifo',{'Uniform',3},{'PER',11},{'PER+IS',12}}};



%% High sampling frequency

% not used
% 40 HSR Swingup 100 Hz, 10k xp, EXPL a1.0, PER+IS
% 41 HSR Swingup 100 Hz, 10k xp, EXPL a2.0, Uniform
% 42 HSR Swingup 100 Hz, 10k xp, EXPL a2.0, PER+IS
% 38 HSR Swingup 100 Hz, 10k xp, FIFO, PER+IS
%-----------------------------------------------------

% 37 HSR Swingup 100 Hz, 10k xp, FIFO, Uniform
% 124 HSR Swingup 100 Hz, 10k xp, FIFO, PER+FIS
% 39 HSR Swingup 100 Hz, 10k xp, EXPL a1.0, Uniform
% 67 HSR Swingup 100 Hz, 10k xp, EXPL a1.0, PER+FIS
% 94 HSR Swingup 100 Hz, 10k xp, TDE a1.0, Uniform
% 68 HSR Swingup 100 Hz, 10k xp, TDE a1.0, PER+FIS
% 95 HSR Swingup 100 Hz, 10k xp, OFFPOL, Uniform
% 96 HSR Swingup 100 Hz, 10k xp, OFFPOL, PER+FIS

% 57 UHSR Magman 200 Hz, 10k xp, FIFO, Uniform
% 61 UHSR Magman 200 Hz, 10k xp, FIFO, PER+FIS
% 58 UHSR Magman 200 Hz, 10k xp, EXPL 1.0, Uniform
% 62 UHSR Magman 200 Hz, 10k xp, EXPL 1.0, PER+FIS
% 123 UHSR Magman 200Hz, 10k xp, TDE 1.0, Uniform
% 69 UHSR Magman 200 Hz, 10k xp, TDE 1.0, PER+FIS
% 97 UHSR Magman 200 Hz, 10k xp, OFFPOL, Uniform
% 98 UHSR Magman 200 Hz, 10k xp, OFFPOL, PER+FIS


%% Noise

% 76 Noise Swingup 0.02, 10k xp, FIFO, Uniform
% 77 Noise Swingup 0.02, 10k xp, FIFO, PER+FIS
% 78 Noise Swingup 0.02, 10k xp, EXPL a1.0, Uniform
% 79 Noise Swingup 0.02, 10k xp, EXPL a1.0, PER+FIS
% 80 Noise Swingup 0.02, 10k xp, TDE a1.0, Uniform
% 81 Noise Swingup 0.02, 10k xp, TDE a1.0, PER+FIS

% 70 Noise Magman 0.02, 10k xp, FIFO, Uniform
% 71 Noise Magman 0.02, 10k xp, FIFO, PER+FIS
% 72 Noise Magman 0.02, 10k xp, EXPL a1.0, Uniform
% 73 Noise Magman 0.02, 10k xp, EXPL a1.0, PER+FIS
% 74 Noise Magman 0.02, 10k xp, TDE a1.0, Uniform
% 75 Noise Magman 0.02, 10k xp, TDE a1.0, PER+FIS


%% Swing up
NUMEXP = 50;
h = shadedrewardplot( {{'FIFO',26},{'TDE',64},{'EXPL',36},{'FULL DB',3}},NUMEXP, 0.90, '', -1 );
axis([0 2000 -1.4 -0.196]);


%% Magman
h = shadedrewardplot( {{'FIFO',13},{'TDE',66},{'EXPL',23},{'FULL DB',1}},NUMEXP, 0.90, '',-1 );
axis([0 2000 -3 -0.2]);

%% SWINGUP Noise
NUMEXP = 50;
h = shadedrewardplot( {{'FIFO',76},{'TDE',80},{'EXPL',78}},NUMEXP, 0.90, '',-1 );
axis([0 3000 -1.4 -0.196]);


%% Magman NOISE
h = shadedrewardplot( {{'FIFO',71},{'TDE',75},{'EXPL',73}},NUMEXP, 0.90, '',-1 );
axis([0 3000 -3 -0.2]);

%% SWINGUP HSR
NUMEXP = 50;
h = shadedrewardplot( {{'FIFO',37},{'TDE',94},{'EXPL',39}},NUMEXP, 0.90, '',-1 );
axis([0 3000 -1.4 -0.196]);


%% Magman HSR
h = shadedrewardplot( {{'FIFO',57},{'TDE',123},{'EXPL',58}},NUMEXP, 0.90, '',-1 );
axis([0 3000 -3 -0.2]);

% 37 HSR Swingup 100 Hz, 10k xp, FIFO, Uniform
% 124 HSR Swingup 100 Hz, 10k xp, FIFO, PER+FIS
% 39 HSR Swingup 100 Hz, 10k xp, EXPL a1.0, Uniform
% 67 HSR Swingup 100 Hz, 10k xp, EXPL a1.0, PER+FIS
% 94 HSR Swingup 100 Hz, 10k xp, TDE a1.0, Uniform
% 68 HSR Swingup 100 Hz, 10k xp, TDE a1.0, PER+FIS
% 95 HSR Swingup 100 Hz, 10k xp, OFFPOL, Uniform
% 96 HSR Swingup 100 Hz, 10k xp, OFFPOL, PER+FIS

% 57 UHSR Magman 200 Hz, 10k xp, FIFO, Uniform
% 61 UHSR Magman 200 Hz, 10k xp, FIFO, PER+FIS
% 58 UHSR Magman 200 Hz, 10k xp, EXPL 1.0, Uniform
% 62 UHSR Magman 200 Hz, 10k xp, EXPL 1.0, PER+FIS
% 123 UHSR Magman 200Hz, 10k xp, TDE 1.0, Uniform
% 69 UHSR Magman 200 Hz, 10k xp, TDE 1.0, PER+FIS
% 97 UHSR Magman 200 Hz, 10k xp, OFFPOL, Uniform
% 98 UHSR Magman 200 Hz, 10k xp, OFFPOL, PER+FIS


%%
%h = shadedrewardplot( {{'P=1',146},{'P=0',150}},50, 0.90, '',-1 );