clear all;
close all;
clc;

magman.standardfull = {{'fifo',{'Uniform',1},{'PER',9},{'PER+IS',10}}};
swingup.standardfull = {{'fifo',{'Uniform',3},{'PER',11},{'PER+IS',12}}};

magman.standard10k = {  {'fifo',{'Uniform',13},{'PER',14},{'PER+IS',15}},...
                        {'E_a025',{'Uniform',16},{'PER',17},{'PER+IS',18}},...
                        {'E_a050',{'Uniform',19},{'PER',20},{'PER+IS',21}},...
                        {'E_a120',{'Uniform',22},{'PER',23},{'PER+IS',24},{'PER+FIS',59}},...
                        {'TDE_a120',{'Uniform',65},{'PER',0},{'PER+IS',0},{'PER+FIS',66}}};
                    
swingup.standard10k = {  {'fifo',{'Uniform',25},{'PER',26},{'PER+IS',27}},...
                        {'E_a025',{'Uniform',28},{'PER',29},{'PER+IS',30}},...
                        {'E_a050',{'Uniform',31},{'PER',32},{'PER+IS',33}},...
                        {'E_a120',{'Uniform',34},{'PER',35},{'PER+IS',36},{'PER+FIS',60}},...
                        {'TDE_a120',{'Uniform',63},{'PER',0},{'PER+IS',0},{'PER+FIS',64}}};                    
                    
magman.highfreq10k = {  {'fifo',{'Uniform',43},{'PER',0},{'PER+IS',44}},...
                        {'E_a100',{'Uniform',45},{'PER',0},{'PER+IS',46}},...
                        {'E_a200',{'Uniform',47},{'PER',0},{'PER+IS',48}}};

magman.uhighfreq10k = { {'fifo',  {'Uniform',57},{'PER',0},{'PER+IS',0},{'PER+FIS',61}},...
                        {'E_a100',{'Uniform',58},{'PER',0},{'PER+IS',0},{'PER+FIS',62}},...
                        {'',{'',0}},{'',{'',0}},...
                         {'TDE_a100',{'Uniform',0},{'PER',0},{'PER+IS',0},{'PER+FIS',69}},...
                         {'OFFPOL1',{'Uniform',97},{'PER',0},{'PER+IS',0},{'PER+FIS',98}}};
                       
                    
swingup.highfreq10k = {  {'fifo',{'Uniform',37},{'PER',0},{'PER+IS',38}},...
                        {'E_a100',{'Uniform',39},{'PER',0},{'PER+IS',40},{'PER+FIS',67}},...
                        {'E_a200',{'Uniform',41},{'PER',0},{'PER+IS',42}},...
                        {'TDE_a100',{'Uniform',94},{'PER',0},{'PER+IS',0},{'PER+FIS',68}},...
                         {'OFFPOL1',{'Uniform',95},{'PER',0},{'PER+IS',0},{'PER+FIS',96}}};
    
magman.highLR10k =   {  {'fifo',{'Uniform',53},{'PER',0},{'PER+IS',54}},...
                        {'E_a100',{'Uniform',55},{'PER',0},{'PER+IS',56}}};                     
                    
swingup.highLR10k =  {  {'fifo',{'Uniform',49},{'PER',0},{'PER+IS',50}},...
                        {'E_a100',{'Uniform',51},{'PER',0},{'PER+IS',52}}};
                    
magman.noise002_10k =   {   {'fifo',{'Uniform',70},{'PER+FIS',71}},...
                            {'E_a100',{'Uniform',72},{'PER+FIS',73}},...
                            {'TDE_a100',{'Uniform',74},{'PER+FIS',75}}};        

swingup.noise002_10k =   {   {'fifo',{'Uniform',76},{'PER+FIS',77}},...
                            {'E_a100',{'Uniform',78},{'PER+FIS',79}},...
                            {'TDE_a100',{'Uniform',80},{'PER+FIS',81}}};              
                        
magman.noise005_10k =   {   {'fifo',{'Uniform',82},{'PER+FIS',83}},...
                            {'E_a100',{'Uniform',84},{'PER+FIS',85}},...
                            {'TDE_a100',{'Uniform',86},{'PER+FIS',87}}};        

swingup.noise005_10k =   {   {'fifo',{'Uniform',88},{'PER+FIS',89}},...
                            {'E_a100',{'Uniform',90},{'PER+FIS',91}},...
                            {'TDE_a100',{'Uniform',92},{'PER+FIS',93}}};  

magman.synth10k = {  {'standard',{'Only real data',13},{'10% Synth S',103},{'10% Synth A',104},{'10% Synth SA',105}},...
                     {'HSR',{'Only real data',57},{'10% Synth S',109},{'10% Synth A',110},{'10% Synth SA',111}},...
                     {'Noise',{'Only real data',70},{'10% Synth S',115},{'10% Synth A',116},{'10% Synth SA',117}}};        
                        
swingup.synth10k = {  {'standard',{'Only real data',25},{'10% Synth S',106},{'10% Synth A',107},{'10% Synth SA',108}},...
                     {'HSR',{'Only real data',37},{'10% Synth S',112},{'10% Synth A',113},{'10% Synth SA',114}},...
                     {'Noise',{'Only real data',76},{'10% Synth S',118},{'10% Synth A',119},{'10% Synth SA',120}}};

                        
                            

%% FULL DB, effect of PER , IS

plot_experiments_rewards( swingup.standardfull, 'Swingup full db ' )
plot_experiments_rewards( magman.standardfull, 'Magman full db' )
%% Smaller DB

plot_experiments_rewards( swingup.standard10k, 'Swingup 10k examples, standard' )
plot_experiments_rewards( magman.standard10k, 'Magman 10k examples, standard' )
%% Samller DB, high sampling frequency

plot_experiments_rewards( swingup.highfreq10k, 'Swingup 10k examples, high sampling frequency' )
plot_experiments_rewards( magman.highfreq10k, 'Magman 10k examples, high sampling frequency (100Hz)' )
plot_experiments_rewards( magman.uhighfreq10k, 'Magman 10k examples, high sampling frequency (200Hz)' )
%% Smaller DB, high learning rate

plot_experiments_rewards( swingup.highLR10k, 'Swingup 10k examples, high learning rate' )
plot_experiments_rewards( magman.highLR10k, 'Magman 10k examples, high learning rate' )


%% Smaller DB, 0.02 noise

plot_experiments_rewards( swingup.noise002_10k, 'Swingup 10k examples, noise 0.02' )
plot_experiments_rewards( magman.noise002_10k, 'Magman 10k examples, noise 0.02' )


%% Smaller DB, 0.05 noise

plot_experiments_rewards( swingup.noise005_10k, 'Swingup 10k examples, noise 0.05' )
plot_experiments_rewards( magman.noise005_10k, 'Magman 10k examples, noise 0.05' )


%% Magman (default) comparisson

plot_experiments_rewards2({ {'Full database',1},{'FIFO-PER+IS',15},{' E1.2 PER+FIS',59},{'TDE 1.2 PER+FIS',66} }, 'Magman 50Hz 10k')
%% Swingup (default) comparisson

plot_experiments_rewards2({ {'Full database, PER+IS',12},{'FIFO-PER+IS',27},{' E1.2 PER+FIS',60},{'TDE 1.2 PER+FIS',64} }, 'Swingup 50Hz 10k')

%% Magman (HSR) comparisson

plot_experiments_rewards2({ {'50 Hz Full database',1},{'200Hz FIFO-PER+FIS',61},{'200Hz E1.0 PER+FIS',62},{'200Hz TDE 1.0 PER+FIS',69} }, 'Magman 200Hz 10k')
%% Swingup (HSR) comparisson

plot_experiments_rewards2({ {'50 Hz Full database PER+IS',12},{'100Hz FIFO-PER+IS',38},{'100Hz E1.0 PER+FIS',67},{'100Hz TDE 1.0 PER+FIS',68} }, 'Swingup 100Hz 10k')
%% NIPS2016

plot_experiments_rewards2({{'FIFO',37},{'OFFPOL',95},{'TDE',94} ,{'Exploration',39} },  'Swingup 100Hz 10k')

%% Synthetic

plot_experiments_rewards(magman.synth10k,'Magman Synthetic samples')
plot_experiments_rewards(swingup.synth10k,'Swingup Synthetic samples')


%%

plot_experiments_rewards2({{'FULLDB + PER+IS',3},{'FIFO + UNIFORM',25}},  'Swingup')
plot_experiments_rewards2({{'FULLDB + PER+IS',1},{'FIFO + UNIFORM',13}},  'Magman')



%%

