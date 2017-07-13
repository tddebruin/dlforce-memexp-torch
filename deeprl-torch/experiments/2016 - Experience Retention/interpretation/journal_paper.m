%% 
%
% Plots and numbers for the journal paper
%
%

%% Synthetic experiences

NUMEXP = 50;

%%

magman.synth10k = {  {'standard',{'Only real data',13},{'10% Synth S',103},{'10% Synth A',104},{'10% Synth SA',105}},...
                     {'HSR',{'Only real data',57},{'10% Synth S',109},{'10% Synth A',110},{'10% Synth SA',111}},...
                     {'Noise',{'Only real data',70},{'10% Synth S',115},{'10% Synth A',116},{'10% Synth SA',117}}};        
                        
swingup.synth10k = {  {'standard',{'Only real data',25},{'10% Synth S',106},{'10% Synth A',107},{'10% Synth SA',108}},...
                     {'HSR',{'Only real data',37},{'10% Synth S',112},{'10% Synth A',113},{'10% Synth SA',114}},...
                     {'Noise',{'Only real data',76},{'10% Synth S',118},{'10% Synth A',119},{'10% Synth SA',120}}};
                 
% SWINGUP

plot_experiments_rewhistogram({{'Only reall data',13},{'States',103},{'Actions',104} ,{'State+Actions',105} },  'Magman',NUMEXP,0.001)
plot_experiments_rewhistogram({{'Only reall data',25},{'States',106},{'Actions',107} ,{'State+Actions',108} },  'Swingup',NUMEXP,0.001)

plot_experiments_rewhistogram({{'Only reall data',13},{'Actions',104} },  'Magman',NUMEXP,0.001)
plot_experiments_rewhistogram({{'Only reall data',25},{'Actions',106} },  'Swingup',NUMEXP,0.001)



plot_experiments_rewhistogram({{'Only reall data',57},{'States',109},{'Actions',110} ,{'State+Actions',111} },  'Magman - HSR',NUMEXP,0.01)
plot_experiments_rewhistogram({{'Only reall data',37},{'States',112},{'Actions',113} ,{'State+Actions',114} },  'Swingup - HSR',NUMEXP,0.01)


plot_experiments_rewhistogram({{'Only reall data',70},{'States',115},{'Actions',116} ,{'State+Actions',117} },  'Magman - Noise',NUMEXP,0.01)
plot_experiments_rewhistogram({{'Only reall data',76},{'States',118},{'Actions',119} ,{'State+Actions',120} },  'Swingup - Noise',NUMEXP,0.01)

%% FULL DB
magman.standardfull = {{'fifo',{'Uniform',1},{'PER',9},{'PER+IS',10}}};
swingup.standardfull = {{'fifo',{'Uniform',3},{'PER',11},{'PER+IS',12}}};



plot_experiments_rewhistogram({{'FIFO',1},{'PER',9},{'PER+IS',10}},  'Magman - FULL DB',50,0.01)
plot_experiments_rewhistogram({{'FIFO',3},{'PER',11},{'PER+IS',12}},  'Swingup - FULL DB',NUMEXP,0.01)








%% Tables, Synthetic data
% Standard
synthdataSU = {{'None',25},{'States',106},{'Actions',107} ,{'State+Actions',108} };
synthdataMM = {{'None',13},{'States',103},{'Actions',104} ,{'State+Actions',105} };
latexcode = journal_table( synthdataSU, synthdataMM, 'Synth samples', NUMEXP, 'Experiments with synthetic data, standard', 'tabsynth', -.4, -0.1972, -0.2063 );

%% High samping rate
synthdataSU = {{'None',37},{'States',112},{'Actions',113} ,{'State+Actions',114},{'None[DE]',121},{'None[DE+$\gamma_c$]',125}, {'States[DE]',130},{'Actions[DE]',131} ,{'State+Actions[DE]',132} };
synthdataMM = {{'None',57},{'States',109},{'Actions',110} ,{'State+Actions',111},{'None[DE]',122},{'None[DE+$\gamma_c$]',126}, {'States[DE]',127},{'Actions[DE]',128} ,{'State+Actions[DE]',129} };
latexcode = journal_table( synthdataSU, synthdataMM, 'Synth samples', NUMEXP, 'Experiments with synthetic data, high sampling rate', 'tabsynth-hsr', -.4, -0.2305, -0.2169 );

%% Noise 
synthdataSU = {{'None',76},{'States',118},{'Actions',119} ,{'State+Actions',120} };
synthdataMM = {{'None',70},{'States',115},{'Actions',116} ,{'State+Actions',117} };
latexcode = journal_table( synthdataSU, synthdataMM, 'Synth samples', NUMEXP, 'Experiments with synthetic data, noise', 'tabsynth-noise', -.4, -0.2391, -0.4049 );

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



% synthdataSU = { {'\method{\fifo}{\unif}',25},{'\method{\fifo}{\per}',26},{'\method{\fifo}{\peris}',27},...
%     {'\method{\expl{0.25}}{\unif}',28},{'\method{\expl{0.25}}{\per}',29},{'\method{\expl{0.25}}{\peris}',30},...
%     {'\method{\expl{0.5}}{\unif}',31},{'\method{\expl{0.5}}{\per}',32},{'\method{\expl{0.5}}{\peris}',33},...
%     {'\method{\expl{1.2}}{\unif}',34},{'\method{\expl{1.2}}{\unif + FIS}',135},{'\method{\expl{1.2}}{\unif + FIS light}',139},{'\method{\expl{1.2}}{\per}',35},{'\method{\expl{1.2}}{\peris}',36},{'\method{\expl{1.2}}{\perfis}',60},{'\method{\expl{1.2}}{\perfis light}',141},....
%     {'\method{\tde{1.2}}{\unif}',63},{'\method{\tde{1.2}}{\unif + FIS}',137},{'\method{\tde{1.2}}{\perfis}',64},{'\method{\tde{1.2}}{\perfis light}',143},...
%     {'\method{\full}{\unif}',3},{'\method{\full}{\per}',11},{'\method{\full}{\peris}',12}};
% 
% synthdataMM = { {'\method{\fifo}{\unif}',13},{'\method{\fifo}{\per}',14},{'\method{\fifo}{\peris}',15},...
%     {'\method{\expl{0.25}}{\unif}',16},{'\method{\expl{0.25}}{\per}',17},{'\method{\expl{0.25}}{\peris}',18},...
%     {'\method{\expl{0.5}}{\unif}',19},{'\method{\expl{0.5}}{\per}',20},{'\method{\expl{0.5}}{\peris}',21},...
%     {'\method{\expl{1.2}}{\unif}',22},{'\method{\expl{1.2}}{\unif + FIS}',136},{'\method{\expl{1.2}}{\unif + FIS light}',140},{'\method{\expl{1.2}}{\per}',23},{'\method{\expl{1.2}}{\peris}',24},{'\method{\expl{1.2}}{\perfis}',59},{'\method{\expl{1.2}}{\perfis light}',142},...
%     {'\method{\tde{1.2}}{\unif}',65},{'\method{\tde{1.2}}{\unif + FIS}',138},{'\method{\tde{1.2}}{\perfis}',66},{'\method{\tde{1.2}}{\perfis light}',144},...
%     {'\method{\full}{\unif}',1},{'\method{\full}{\per}',9},{'\method{\full}{\peris}',10}};


synthdataSU = { {'\method{\fifo}{\unif}',25},{'\method{\fifo}{\per}',26},{'\method{\fifo}{\peris}',27},...
    {'\method{\expl{0.25}}{\unif}',28},{'\method{\expl{0.25}}{\per}',29},{'\method{\expl{0.25}}{\peris}',30},...
    {'\method{\expl{0.5}}{\unif}',31},{'\method{\expl{0.5}}{\per}',32},{'\method{\expl{0.5}}{\peris}',33},...
    {'\method{\expl{1.2}}{\unif}',34},{'\method{\expl{1.2}}{\unif + FIS}',135},{'\method{\expl{1.2}}{\per}',35},{'\method{\expl{1.2}}{\peris}',36},{'\method{\expl{1.2}}{\perfis}',60},....
    {'\method{\tde{1.2}}{\unif}',63},{'\method{\tde{1.2}}{\unif + FIS}',137},{'\method{\tde{1.2}}{\perfis}',64},...
    {'\method{\full}{\unif}',3},{'\method{\full}{\per}',11},{'\method{\full}{\peris}',12}};

synthdataMM = { {'\method{\fifo}{\unif}',13},{'\method{\fifo}{\per}',14},{'\method{\fifo}{\peris}',15},...
    {'\method{\expl{0.25}}{\unif}',16},{'\method{\expl{0.25}}{\per}',17},{'\method{\expl{0.25}}{\peris}',18},...
    {'\method{\expl{0.5}}{\unif}',19},{'\method{\expl{0.5}}{\per}',20},{'\method{\expl{0.5}}{\peris}',21},...
    {'\method{\expl{1.2}}{\unif}',22},{'\method{\expl{1.2}}{\unif + FIS}',136},{'\method{\expl{1.2}}{\per}',23},{'\method{\expl{1.2}}{\peris}',24},{'\method{\expl{1.2}}{\perfis}',59},...
    {'\method{\tde{1.2}}{\unif}',65},{'\method{\tde{1.2}}{\unif + FIS}',138},{'\method{\tde{1.2}}{\perfis}',66},...
    {'\method{\full}{\unif}',1},{'\method{\full}{\per}',9},{'\method{\full}{\peris}',10}};


%latexcode = numbers_experiments_rewhistogram( synthdataSU, synthdataMM, 'method', NUMEXP, 'Standard conditons - exploration based experience retention', 'tabexpl', -.4 );
latexcode = journal_table( synthdataSU, synthdataMM, 'method', NUMEXP, 'Standard conditons - exploration based experience retention', 'tabexpl', -.4, -0.1972, -0.2063);

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

dataSU = { {'\method{\fifo}{\unif}',37},{'\method{\fifo}{\perfis}',124},...
    {'\method{\expl{1.0}}{\unif}',39},{'\method{\expl{1.0}}{\perfis}',67},{'\method{\expl{1.0}}{\perfis}[DE]',134},...
    {'\method{\tde{1.0}}{\unif}',94},{'\method{\tde{1.0}}{\perfis}',68},...
    %{'\method{\offpol}{\unif}',95},{'\method{\offpol}{\perfis}',96},....
    };

dataMM = { {'\method{\fifo}{\unif}',57},{'\method{\fifo}{\perfis}',61},...
    {'\method{\expl{1.0}}{\unif}',58},{'\method{\expl{1.0}}{\perfis}',62},{'\method{\expl{1.0}}{\perfis}[DE]',133},...
    {'\method{\tde{1.0}}{\unif}',123},{'\method{\tde{1.0}}{\perfis}',69},...
    %{'\method{\offpol}{\unif}',97},{'\method{\offpol}{\perfis}',98},....
    };

latexcode = journal_table( dataSU, dataMM, 'method', NUMEXP, 'High sampling frequencies, method performance', 'tabhsr', -.4,  -0.2305, -0.2169);

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

dataSU = { {'\method{\fifo}{\unif}',76},{'\method{\fifo}{\perfis}',77},...
    {'\method{\expl{1.0}}{\unif}',78},{'\method{\expl{1.0}}{\perfis}',79},...
    {'\method{\tde{1.0}}{\unif}',80},{'\method{\tde{1.0}}{\perfis}',81},...
    };

dataMM = { {'\method{\fifo}{\unif}',70},{'\method{\fifo}{\perfis}',71},...
    {'\method{\expl{1.0}}{\unif}',72},{'\method{\expl{1.0}}{\perfis}',73},...
    {'\method{\tde{1.0}}{\unif}',74},{'\method{\tde{1.0}}{\perfis}',75},...
    };

latexcode = journal_table( dataSU, dataMM, 'method', NUMEXP, 'Sensor and actuator noise, method performance', 'tabnoise', -.4, -0.2391,-0.4049 );



%% figure motivation


%% Swing up
h = shadedrewardplot( {{'FULL DB[PER+IS]',3},{'FIFO[Uniform]',25}},NUMEXP, 0.99, '', -0.1969 );
axis([0 2000 -1.4 0]);


%% Magman
h = shadedrewardplot( {{'FULL DB[PER+IS]',1},{'FIFO[Uniform]',13}},NUMEXP, 0.99, '',-0.2063 );
axis([0 2000 -3 0]);




