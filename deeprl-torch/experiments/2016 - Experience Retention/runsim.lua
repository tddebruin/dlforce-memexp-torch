require 'os'
package.path = "../../lib/?.lua;" .. package.path
require 'drl_communicatorPB'
require 'drl_experience_database'
require 'drl_experience_classifier'
require 'drl_policy'
require 'drl_dqn'
require 'drl_ddpg'
require 'drl_diagnostics'
require 'drl_daydream'
require 'drl_daydream_modelcheck'
--require 'mattorch'
--display = require 'display' -- th -ldisplay.start
--https://github.com/szym/display



USEGPU = false
LOGQPERF = true -- NEEDS TO BE TRUE FOR PRIORITIZED EXPERIENCE REPLAY!


function setup()

	env_scenario = nil

	plots = {
						reward = false,
						updateperiod = 5
					}

	-- settings through command line
	local cmd = torch.CmdLine()
	-- note torch cant deal with passing booleans, therefore these options are false unless the program is called with their flags and no arguments.

	-- Software details
	cmd:option('-noexec',false, 'This module does not interact with the environment.')
	cmd:option('-notrain',false, 'This module does not train the networks.')
	cmd:option('-deterministic',false,'Use a fixed seed.')
	cmd:option('-resultfile', './data/CartPole.res','The path and name of the file where the results will be stored.')

	-- Experiment settings
	cmd:option('-env','SwingupSimC') -- MagmanSimC
	cmd:option('-episodes',2000)
	cmd:option('-samplefreq',50,'Hz')
	cmd:option('-seqlength',4,'seconds')
	cmd:option('-noisescale',0.00,'Multiplies gaussian noise with stdv 1 and adds the noise to state and actions ')
	cmd:option('-bwd',false,'use bwd instead of actual velocity')
	cmd:option('-altrew',false,'use rew func without speed penalty')
	cmd:option('-OUSR',1000,'OU Sampling rate (1000 = base)')

	cmd:option('-min_exploration', 0.1, 'minimum exploration')

	-- Experience replay
	cmd:option('-xpmsize',10000,'Maximum number of experiences in memory') -- 50000
	cmd:option('-overwrite','FIFO','Overwrite policy for the experience database: FIFO, TDE, DISTRIBUTION HYBRID')
	cmd:option('-overwrite_alpha',0.5,'alpha parameter of the overwrite policy')
	cmd:option('-overwrite_metric','OFFPOL','Metric to base the overwrite policy on')
	cmd:option('-hybridrate',1)
	cmd:option('-prioritized_experience_replay',false,'Use rank based prioritized experience replay')
	cmd:option('-prioritized_alpha',0.7,'Alpha constant for rank based probabalistic experience replay. 0 = uniform, 1 = strongly rank based, 0.7 = per paper')
	cmd:option('-prioritized_beta_0',0.5,'Initial prioritized experience replay imprtance sampling constant (0 = no IS, 1 = full correction)')
	cmd:option('-prioritized_beta_final',1,'Final prioritized experience replay imprtance sampling constant (0 = no IS, 1 = full correction) (linear scaling per episode_')
	cmd:option('-countbasedimpsamp',false,'Base the importance sampling on the number of times an experience was used')
	cmd:option('-ignorefrac',0.0,'Fraction of the experiences not to be written to the database')

	cmd:option('-logusecount',true,'track how often each experience is used')
	cmd:option('-savedbsnaps',false,'save snapshots of the state of the database')

	-- Synthetic experiences
	cmd:option('-synthS',false,'Add synthetic state experiences')
	cmd:option('-synthA',false,'Use uniform actions for the experiences')
	cmd:option('-synthFrac',0.0,'Fraction of experiences that are replaced with synthetic experiences')
	cmd:option('-synthRefreshProb',1.0,'probability of refreshing a synthetic experience, as opposed to keeping it the same when the experience is updated. Also keeps the db indices for synth exp fixed when smaller than 1.')

	-- Generalization
	cmd:option('-generalizationrun', false, 'perform a generalization run after training, starting from another initial state distributoin')

	-- RL SPECIFIC
	cmd:option('-gamma',0.95,'')
	cmd:option('-immediate_reward_scale',1)
	-- DL SPECIFIC
	cmd:option('-batchnorm',false,'')
	cmd:option('-batchsize',16,'')
	cmd:option('-samplereuse',8,'Use each sample approx this many times')
	cmd:option('-doubleQ',false)
	cmd:option('-bignet', false, 'Use a network the size of the original DDPG paper instead of a smaller one.')
	-- Critic only specific
	cmd:option('-lr',0.00025,'')
	cmd:option('-l2',0.00001,'')
	cmd:option('-freezecount', 200, 'number of steps between DQN frozen net updates')
	-- Actor-critic specific
	cmd:option('-criticl2',0.005,'')
	cmd:option('-lowpass',1e-2,'')
	cmd:option('-lractor',1e-4,'')
	cmd:option('-lrcritic',1e-3,'')


	opt = cmd:parse(arg)
	opt.exec 	= not(opt.noexec)
	opt.train = not(opt.notrain)
	opt.batchupdates = math.floor((1 - opt.ignorefrac) * opt.seqlength * opt.samplefreq * opt.samplereuse / opt.batchsize)
	if not opt.deterministic then
		math.randomseed( os.time() )
		torch.seed()
	end

	EPISODES = opt.episodes
	CONTINUOUSACTIONS = (opt.env == 'Pendulum-v0' or opt.env == 'BipedalWalker-v2' or opt.env == 'MagmanSimC' or opt.env == 'SwingupSimC')


------------------ Communicator ----------------------------------------------
-- Getting states and/or observations and rewards. Sending actions. This part takes care of abstracting the communication details away so that the rest of the code stays constant for different benchmarks.

	if not communicator then
		communicator 		= communicatorPB({sampling_time = -1},true)

		if opt.env == 'MagmanSimD' or opt.env == 'MagmanSimC' or opt.env == 'SwingupSimD' or opt.env == 'SwingupSimC' then
			require 'drl_experiment' -- includes the lua version of the swingup and magman code.
			local experiment_def = {}
			if opt.env == 'MagmanSimD' then
				experiment_def = {environment = 'magman', magnets = 3, discrete = true, ts = 1/opt.samplefreq}
			elseif opt.env == 'SwingupSimD' then
				experiment_def = {environment = 'pendulum', discrete = true, ts = 1/opt.samplefreq, bwddiff=opt.bwd, altrew = opt.altrew}
			elseif opt.env == 'MagmanSimC' then
				experiment_def = {environment = 'magman', magnets = 4, discrete = false, ts = 1/opt.samplefreq}
			elseif opt.env == 'SwingupSimC' then
				experiment_def = {environment = 'pendulum', discrete = false, ts = 1/opt.samplefreq, bwddiff=opt.bwd, altrew = opt.altrew}
			end
			local channelsettings = {
				internal  	= true,
				name 				= "benchmark",
				experiment 	= drl_experiment(experiment_def)
			}
			channel 				= communicator:add_channel(channelsettings)
			localexperiment = drl_experiment(experiment_def)
		else
			channel 				= communicator:add_channel(
				{
					name 			= "openaigym",
					bindpub 	= "tcp://*:5555",
					bindsub		= "tcp://*:5556"
				})
			communicator.envcommand = 'gnome-terminal -x bash -c "python ../gymzmqmin.py ' .. opt.env .. '"'
		end
	end


	communicator:reset_clock()
	communicator:cleanstart()

------------------ Experience Memory -----------------------------------------
--[[ Used to collect the experience tuples and observations ]]--

	em_settings = {
		full_state_dimension 		= {channel.statedimension},
		observation_dimension 	= nil, -- multimodal example: for a 50x50 rgbd image where the d is seperate:
		--{torch.LongStorage({50,50,3}) , torch.LongStorage({50,50}}
		short_term_memory_size 	= 5000,
		experience_replay_size 	= opt.xpmsize,
		overwrite_policy 				= opt.overwrite or ' FIFO',--'DISTRIBUTION',
		overwrite_alpha	 				= opt.overwrite_alpha,
		overwrite_metric 				= opt.overwrite_metric,
		extra_info 							= {
			{
				name = "TDE",
				default_value = math.huge,
			},
			{
				name = "OFFPOL",
				default_value = -1,
			},
			{
				name = "STATEDESIRABILITY",
				default_value = math.huge,
			},
			{
				name = "STATENOISE",
				default_value = 0,
			},
			{
				name = "ACTIONNOISE",
				default_value = 0,
			},
      {
        name = "UPDATE", -- last nn update for which this datapoint has been used
        default_value = 0,
      },
      {
        name = "USECOUNT", -- number of times a sample has been used in a batch
        default_value = 0,
      },
      {
        name = "QPRED", --
        default_value = 0,
      },
      {
        name = "QFPRED", --
        default_value = 0,
      },
		},
		RL_state_parts 	= {
			observation = false,
			state 			= true,
			action 			= true,
			next_state	= true,
			reward 			= true,
			terminal 		= true,
			next_action = false
		},
		action_dimension 	= {channel.actiondimension},
		get_OSAR_function = function(self,time_index)
			return -- Empty function for the policy executor
		end,
		send_OSAR_function = function(self,time_index)
			return -- Empty function for the training function
		end,
	}

  xpm 						= experience_memory(true,true,em_settings)
  xpm_learn 			= xpm

  -- no factorial in torch or lua?
  local Y = function (g)
	  local a = function (f) return f(f) end
	  return a(function (f)
	     return g(function (x)
	         local c=f(f)
	         return c(x)
	       end)
	   end)
  end

  -- factorial without recursion
  F = function (f)
      return function (n)
         if n == 0 then return 1
         else return n*f(n-1) end
  	     end
	    end

  factorial = Y(F)   -- factorial is the fixed point of F


  xpm.bintab = {}
  xpm.bintab[0] = 1
	local epsinmem = math.floor(opt.xpmsize / ((1-opt.ignorefrac)*(opt.samplefreq * opt.seqlength)))
	local sampprob = opt.samplereuse / epsinmem
	local prob_sampled_i_times = function ( i )
		if i > epsinmem then
			return 0
		else
			return (factorial(epsinmem)/(factorial(i)*factorial(epsinmem-(i)))*sampprob^(i) * (1-sampprob)^(epsinmem-(i)))
		end
	end
	for i = 1,20 do
		xpm.bintab[i] = xpm.bintab[i-1] - prob_sampled_i_times(i)
	end



  if opt.overwrite == 'OFFPOL' then
		POLDISTcriterion = nn.AbsCriterion()
		POLdistances = torch.Tensor(em_settings.experience_replay_size):fill(math.huge)
	  xpm.distribution_index = function()
			local minpold, index
			minpold, index = torch.min(POLdistances,1)
			POLdistances[index[1]] = math.huge
			return index[1]
		end
		xpm.updateDistributions = function ()
			local a 		= xpm.experience_database.action[1]
			local pola 	= network.train_networks.actor.network:forward(xpm.experience_database.state[1])
			local delta = pola:add(-1,a)
			delta:pow(2)
			POLdistances:copy(torch.sum(delta,2))
		end
	end





---------------- Neural networks -------------------------------------------


	if CONTINUOUSACTIONS then
		if opt.bignet then
			network_definitions  =	{
				statesize 		= channel.statedimension[1],
				actionsize 		=	channel.actiondimension[1],
	      state_bounds  = nil,
	      --action_bounds = torch.Tensor({channel.action_bounds.min,channel.action_bounds.max}),
	      actor					=
					{
						hsizes 			= {400,300}, -- 25 25
						nonlinearity 	= nn.ReLu,
						batchnorm 		= opt.batchnorm,
					},
				critic 				=
					{
						actionlayer  	= 1,
						hsizes 				= {400,300}, -- 25 25 10
						nonlinearity 	= nn.ReLu,
						batchnorm 		= false,
						knownQ				= {
							use 				= false, -- use a batch for batchnorm
						},
					},
				GPU						= USEGPU, -- for the training networks
			}
			network = drl_ddpg(opt.exec,opt.train, network_definitions)
		else
			network_definitions  =	{
				statesize 		= channel.statedimension[1],
				actionsize 		=	channel.actiondimension[1],
	      state_bounds  = nil,
	      --action_bounds = torch.Tensor({channel.action_bounds.min,channel.action_bounds.max}),
	      actor					=
					{
						hsizes 			= {50,50}, -- 25 25
						nonlinearity 	= nn.ReLu,
						batchnorm 		= opt.batchnorm,
					},
				critic 				=
					{
						actionlayer  	= 1,
						hsizes 				= {50,50,20}, -- 25 25 10
						nonlinearity 	= nn.ReLu,
						batchnorm 		= false,
						knownQ				= {
							use 				= false, -- use a batch for batchnorm
						},
					},
				GPU						= USEGPU, -- for the training networks
			}
			network = drl_ddpg(opt.exec,opt.train, network_definitions)
		end

	else
		network_definitions  =	{
				statesize 		= channel.statedimension,
				actionsize 		=	channel.actiondimension,
	      state_bounds  = nil,
	      action_bounds = nil,
				hsizes 				= {50,50,20}, -- 25 25
				nonlinearity 	= nn.ReLu,
				batchnorm 		= false,
				GPU						= USEGPU, -- for the training networks
		}
		network = drl_dqn(opt.exec,opt.train, network_definitions)

	end

	network_train_settings =
	{
 			logTDE				= LOGQPERF,
 			logUPDATE     = LOGQPERF,
      logQPred      = LOGQPERF,
      logQFpred     = LOGQPERF,
      logUseCount 	= opt.logusecount,
      batch_size		= opt.batchsize,
 			batchupdates 	= opt.batchupdates,-- 20--math.floor((8/64)*EPISODE_SIZE),
 			gamma 				= opt.gamma,
 			L2norm				= opt.l2,
 			prioritized_experience_replay = opt.prioritized_experience_replay,
 			prioritized_alpha 						= opt.prioritized_alpha,
 			countbasedimpsamp 						= opt.countbasedimpsamp,
 			optimsettings = {
 				optimfunction 	= optim.adam,
 				config 					= {learningRate = opt.lr}, --1e-3
 				freezecount 		= opt.freezecount,
 				configActor 		= {learningRate = opt.lractor}, --1e-3
 				configCritic 		= {learningRate = opt.lrcritic},  -- 1e-4
 				critic_L2norm		= opt.l2,
 				targetlowpass 	= opt.lowpass,
 			}
	}



----------------- Policy ----------------------------------------------------
	if CONTINUOUSACTIONS then
		--local explorationPolicy  	= drl_continuous_policy_normal_random({action_dimension = channel.actiondimension, lower_limits = {-1, -1, -1, -1}, upper_limits = {1, 1, 1, 1}})
		--local explorationPolicy  	= drl_continuous_policy_uniform_random({action_dimension = channel.actiondimension, lower_limits = channel.action_bounds.min, upper_limits = channel.action_bounds.max})
		local explorationPolicy  	= drl_policy_OU({theta = 5.14, sigma = 0.3, sample_rate = opt.OUSR, action_dimension = channel.actiondimension, bounds = {-1, 1}}) -- DDPG paper theta+sigma, sr=1000: no subsampling

		policy 													= drl_policy({action_dimension = channel.actiondimension,
			 	communicator 								= communicator,
			 	exploitationActionFunction 	= function (state) return network:get_policy_action(state) end,
			 	explorationActionFunction		= explorationPolicy,
			 	tradeofftype								= 'add',--'addunscaled', -- scale
			 	bounds										= {-1,1}, --! All the same!
			 	explorationAmountFunction		= drl_exploration_amount_function({
			 		functiontype 								= 'linear_per_sequence',
			 		initial_exploration  				= 1.00,--1.2
					multiplier					    	= 0.9/500, -- 0.999,
			 		minimum_exploration					= opt.min_exploration--0.6,
	 			}),
		})
	else
		local explorationPolicy  = drl_discrete_policy_uniform_random({action_dimension = channel.actiondimension})
		policy 													= drl_policy({action_dimension = channel.actiondimension,
			 	communicator 								= communicator,
			 	exploitationActionFunction 	= function (state) return network:get_policy_action(state) end,
			 	explorationActionFunction		= explorationPolicy,
			 	tradeofftype								= 'greedy',
			 	explorationAmountFunction		= drl_exploration_amount_function({
			 		functiontype 								= 'linear_per_sequence',
			 		initial_exploration  				= 0.70,
			 		multiplier						    	= 0.69/500, -- 0.999,
			 		minimum_exploration					= 0.01--0.6,
	 			}),
		})
	end


---------------- Experiment description ------------------------------------
	createNormalizationTable = function ()
		normalization = {}
		if channel.action_bounds.max and CONTINUOUSACTIONS then
			local amax = torch.Tensor({channel.action_bounds.max})
			local amin = torch.Tensor({channel.action_bounds.min})
			normalization.output_scale 	= amax:clone():csub(amin):mul(0.5)
			normalization.output_add 		= amax:clone():add(amin):mul(0.5)
		end
		if torch.isTensor(channel.state_bounds.max) then
			local smax = channel.state_bounds.max
			local smin = channel.state_bounds.min
			normalization.input_scale 	= smax:clone():csub(smin):mul(0.5)
			normalization.input_add 		= smax:clone():add(smin):mul(0.5)
		end
		return
	end

	getStateAddNoiseAndNormalize = function (time_index, noNoise)
		channel:receive_n_fortimeindex(2,time_index)
		local recstate 	= channel:get_state_part(time_index,'state','state')
		local normstate = recstate
		if not(normalization) then
			createNormalizationTable()
		end

		if normalization.input_scale then
			normstate:add(-1,normalization.input_add)
			normstate:cdiv(normalization.input_scale)
		else
			assert(fasle)
		end

		if opt.noisescale > 0 then
			statenoise = opt.noisescale*torch.randn(normstate:size())
			if statenoise:nElement() == 1 then
				LAST_STATE_NOISE = statenoise:clone()
			else
				LAST_STATE_NOISE = statenoise:clone():norm()
			end

			normstate:add(statenoise)
		end

		return normstate
	end

	scaleActionBeforeSend = function(action)
		if not(normalization) then
			createNormalizationTable()
		end
		if normalization.output_scale then
			action:cmul(normalization.output_scale)
			action:add(normalization.output_add)
		end
		return action
	end



	gymstate = generalized_state({
		name = 'Gym State',
		experience_memory = xpm,
		dimension = channel.statedimension,
		extrap = "nil",
		statetype = "torch.DoubleTensor",
		prevPerFullState = 0,
		getFunction = function(self,time_index)
				return getStateAddNoiseAndNormalize(time_index)
		end
	})



	action = generalized_state({
		name = 'action',
		experience_memory = xpm,
		dimension = channel.actiondimension,
		extrap = "zero",
		statetype = "torch.DoubleTensor",
		getFunction 	= function(self,time_index,full_state)
			local unscalecdaction = policy(full_state[1],time_index,communicator:get_sequence_index())
			local actuatoraction  = unscalecdaction:clone()
			if opt.noisescale > 0 then
				--actuatoraction:add(opt.noisescale*torch.randn(unscalecdaction:size()))
				actionnoise = opt.noisescale*torch.randn(unscalecdaction:size())
				if actionnoise:nElement() == 1 then
					LAST_ACTION_NOISE = actionnoise:clone()
				else
					LAST_ACTION_NOISE = actionnoise:clone():norm()
				end
				actuatoraction:add(actionnoise)
			end
			local action 	= scaleActionBeforeSend(actuatoraction)

      if (channel:get_terminal(time_index) == 1) then
				channel:send_env_reset(env_scenario)
				--communicator:sleep(0.2)
			else
      	local send_action
      	if action:nElement() > 1 and (not( CONTINUOUSACTIONS)) then
      		local av,ai = torch.max(action,1)
      		send_action = torch.Tensor({ai[1]})
      	else
      		send_action = action
      	end
      	channel:send_action("action",send_action)
      end
      return unscalecdaction
		end
	})

	reward = generalized_state({
		name = 'reward',
		experience_memory = xpm,
		dimension = torch.LongStorage({1}),
		extrap = "nil",
		statetype = "torch.DoubleTensor",
		getFunction = function(self,time_index)
			return opt.immediate_reward_scale * channel:get_reward(time_index)
		end
	})

	terminal = generalized_state({
		name = 'terminal',
		experience_memory = xpm,
		dimension = torch.LongStorage({1}),
		extrap = "nil",
		statetype = "torch.DoubleTensor",
		getFunction = function(self,time_index)
			if channel:get_terminal(time_index) == 1 then
				return torch.Tensor({1})
			else
				return torch.Tensor({0})
			end
		end
	})

	xpm:reset_getfunctions()
	xpm:addPartialState(1,gymstate)
	xpm:setAction(action)
	xpm:setReward(reward,{delay=true,cost=false}) -- calculated reward is for the previous state action pair (based on s,a,s'), reward is infact a cost (minimize instead of maximize)
	xpm:setTerminal(terminal) -- for environments that have absorbing states

	diagnostics = drl_diagnostics({})
	--diagnostics:addConsoleSummary(reward,{})

	print('Setup complete')
end

function replace_last_experience_with_fantasy( probability )
	if probability > 0 then
			assert(opt.overwrite == 'FIFO')
	end
	last_index = xpm.experience_database.current_write_index - 1
	if last_index > 0 then
		if math.random() < probability then
			if opt.synthS then
				xpm.experience_database.state[1][last_index]:uniform(-1,1)
			end
			if opt.synthA then
				xpm.experience_database.action[1][last_index]:uniform(-1,1)
			else
				xpm.experience_database.action[1][last_index]:copy(policy(xpm.experience_database.state[1][last_index],communicator:get_time_index(),communicator:get_sequence_index()))
			end
			local actuatoraction  = xpm.experience_database.action[1][last_index]:clone()
			if opt.noisescale > 0 then
				actuatoraction:add(opt.noisescale*torch.randn(actuatoraction:size()))
			end
			local action 	= scaleActionBeforeSend(actuatoraction)

    	local s1 = xpm.experience_database.state[1][last_index]:clone()
    	if opt.noisescale > 0 then
    		s1:add(opt.noisescale*torch.randn(s1:size()))
    	end
			localexperiment:set_state(s1,true)
			localexperiment:step(action)

			local s2 = localexperiment:get_state()

			if not(normalization) then
				createNormalizationTable()
			end

			if normalization.input_scale then
				s2:add(-1,normalization.input_add)
				s2:cdiv(normalization.input_scale)
			else
				assert(fasle)
			end

			if opt.noisescale > 0 then
				s2:add(opt.noisescale*torch.randn(s2:size()))
			end

			xpm.experience_database.next_state[1][last_index]:copy(s2)
			xpm.experience_database.reward[last_index] = opt.immediate_reward_scale * localexperiment:get_reward()
  	end
	end
end

function generalization_experiment()

	local result = torch.Tensor(40):zero()
	for experiment_idx = 1,40 do
		localexperiment:reset(experiment_idx)
		local rsum = 0
		local s = localexperiment:get_state():clone()
		s:add(-1,normalization.input_add)
		s:cdiv(normalization.input_scale)
		local a = scaleActionBeforeSend(policy:deterministic(torch.Tensor(s)))
		for ts = 1,200 do
			localexperiment:step(torch.Tensor(a))
			s = localexperiment:get_state():clone()
			s:add(-1,normalization.input_add)
			s:cdiv(normalization.input_scale)
			rsum = rsum + localexperiment:get_reward()
			a = scaleActionBeforeSend(policy:deterministic(torch.Tensor(s)))
		end
		result[experiment_idx] = rsum/(opt.seqlength * opt.samplefreq/10)
	end
	return result
end


function recalculate_bintab( episode )
	if opt.countbasedimpsamp then
		local epsinmem = math.floor(opt.xpmsize / ((1-opt.ignorefrac)*(opt.samplefreq * opt.seqlength)))
		if episode > epsinmem then
			return -- no need to keep calculating, is fixed now
		end
		local n = math.min(episode, epsinmem)
		local p = opt.samplereuse / epsinmem
		local prob_sampled_i_times = function ( i )
			if i > epsinmem then
				return 0
			else
				return (factorial(n)/(factorial(i)*factorial(n-(i)))*p^(i) * (1-p)^(n-(i)))
			end
		end

		for i = 1,20 do
			s = 0
			for j=0,i do
				if j <= n then
					s = s + prob_sampled_i_times(j)
				end
			end
			xpm.bintab[i] = 1 - s
		end
		expected_replays = math.ceil(n*p)
		cum_weight = 0
		for j=1,expected_replays do
			cum_weight = cum_weight + xpm.bintab[j]
		end
		correction_weight = expected_replays / cum_weight
		for i = 1,20 do
			xpm.bintab[i] = xpm.bintab[i] * correction_weight
			if xpm.bintab[i] ~= xpm.bintab[i] then
				 xpm.bintab[i] = 1
			end
		end
	end
end






function main()
	setup()

	LAST_STATE_NOISE, LAST_ACTION_NOISE = 0,0

	if opt.synthRefreshProb < 1.0 then
		xpm.synt_indices = torch.Tensor(xpm.experience_replay_size)
		for i=1, xpm.experience_replay_size do
			if math.random() < opt.synthFrac then
				xpm.synt_indices[i] = 1
			else
				xpm.synt_indices[i] = 0
			end
		end
	end

	rewards = torch.Tensor(EPISODES):zero()
	gen_rewards = torch.Tensor(50, 40):zero()
	channel:send_env_reset(env_scenario)
	communicator:advance_clock()
	local BESTSOFAR = -math.huge
	local sequence_timestep = 0
	recalculate_bintab(1)
	-- main loop -----------------------
	sprintf('starting main loop')
	while communicator:get_sequence_index() <= EPISODES do
		time_index 			= communicator:get_time_index() -- always increasing timestep counter for keeping track of all events
		sequence_index 	= communicator:get_sequence_index() -- counter for the episode
		sequence_timestep = sequence_timestep + 1 -- timesteps since the beginning of the current episode

		xpm:collect_OSAR(time_index,sequence_index) -- collect the (observations) state action and reward by interacting with the environment

		if opt.overwrite == 'HYBRID' then
			xpm_offpol.OSAR = xpm:get_OSAR_copy()
			xpm_offpol:add_RL_state_to_db()
		end
		if opt.synthRefreshProb < 1.0 then
			idx = xpm.experience_database.current_write_index
			if xpm.synt_indices[idx] == 1 then
				local dbidx = xpm:dont_add_RL_state_to_db()
				if dbidx then
					if math.random() < opt.synthRefreshProb or dbidx < xpm.experience_replay_size then
						replace_last_experience_with_fantasy( 1.0 )
					end
				end
			else
				local dbidx = xpm:add_RL_state_to_db()
			end
			assert(opt.ignorefrac==0.0)
		else
			local dbidx = xpm:add_RL_state_to_db()
			if dbidx then
				replace_last_experience_with_fantasy( opt.synthFrac )
				xpm:update_extra_info("OFFPOL",policy:get_exploration_effect(),dbidx)
				if opt.noisescale > 0.0 then
					--xpm:update_extra_info("STATENOISE",LAST_STATE_NOISE:clone(),dbidx)
					--xpm:update_extra_info("ACTIONNOISE",LAST_ACTION_NOISE:clone(),dbidx)
				end
				if opt.ignorefrac > 0 and math.random() < opt.ignorefrac then
					xpm:rewind_overwrite_index() -- next time overwrite this experience
				end
				if opt.overwrite ==  'RESERVOIR' and xpm.experience_database.last_write_index == xpm.experience_replay_size then
					keep_chance = xpm.experience_replay_size / time_index
					if math.random() > keep_chance then
						xpm:rewind_overwrite_index()
					end
				end
			end
		end


		diagnostics:update(time_index, sequence_index)
		if channel:get_terminal(time_index) == 1 or sequence_timestep >= (opt.seqlength * opt.samplefreq) then
			if sequence_timestep >= (opt.seqlength * opt.samplefreq) then
				channel:send_env_reset(env_scenario)
				sequence_timestep = 0
			end
			if opt.overwrite == 'HYBRID' then
				fill_hybrid()
			end

			-- prioritized experience replay imprtance sampling annealing
			network_train_settings.prioritized_beta = opt.prioritized_beta_0 + (opt.prioritized_beta_final - opt.prioritized_beta_0) * sequence_index/EPISODES
			network:train(xpm_learn, network_train_settings)

    	network:set_controller_parameters(network:get_controller_parameters())
			local seqrew = reward:ordered_seq(sequence_index):sum()/(opt.immediate_reward_scale * opt.seqlength * opt.samplefreq/10)

			if sequence_index%40 == 1 then
				print("Sequence " .. sequence_index .. " / " .. EPISODES)
				print("TDE for last update: " .. calculate_TDE(xpm ,{since="last_update"}))
	 			print("AVG pred Q for last update: " .. calculate_AVGQ(xpm ,{since="last_update"})[1] .. ", QFrozen: " .. calculate_AVGQ(xpm ,{since="last_update"})[2])
	    	print("DB filled: " .. math.floor(100*xpm.experience_database.last_write_index/xpm.experience_replay_size) .. "%")
				print(seqrew)
				if opt.savedbsnaps then
					xpm:save_mat_screenshot('test' .. sequence_index .. '.mat',{time_indices = true, sequence_indices = true, state = true, action = true, reward = true},{'TDE','USECOUNT','OFFPOL'})
				end
				if opt.generalizationrun then
					gen_rewards[1+ math.floor(sequence_index/40)] = generalization_experiment()
					print(gen_rewards[1+ math.floor(sequence_index/40)])
				end
			end
			rewards[sequence_index] = seqrew
			communicator:advance_sequence_index()
			recalculate_bintab(communicator:get_sequence_index())
		end
		-- advancing the clock increases the time index by one, sends it out on all channels and then blocks until at least one sampling time period has passed since the last time the clock advancement function call was completed.
		communicator:advance_clock()
	end

end

main()
torch.save(opt.resultfile .. '-t7',{seq = rewards})
if opt.generalizationrun then
	print('Generalization: ')
	print(torch.mean(gen_rewards))
	torch.save(opt.resultfile .. '_gen-t7', {seq = gen_rewards})
end
--mattorch.save(opt.resultfile,{seq = rewards})
--mattorch.save(opt.resultfile,{seq = rewards, epsToComp = torch.Tensor({sequence_index})})
torch.save(opt.resultfile .. 'completed') -- to signal the experiment is done
print("READY")
channel:send_exit()
communicator:close()

os.execute("sleep 2")
