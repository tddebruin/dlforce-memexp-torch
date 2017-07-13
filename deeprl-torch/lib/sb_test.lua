--require('mobdebug').start()

require 'exec_switchboard'
require 'experiment_settings'
require 'communicatorPB'
require 'experience_database'
require 'drl_policy' 
require 'drl_ddpg'
require 'drl_diagnostics'

-- TODO: Reward and reference to xy
-- TODO: THOU SHALT NOT HARDCODE DIMENSIONS!
-- TODO: Settings file
-- TODO: Single PC multi process
-- TODO: Multi PC
-- TODO: Setup drivers
-- TODO: Matlab protobuf
-- TODO: Documentation
-- TODO: Diagnostics: signal  - avg per sequence
			-- Diagnoostics: signal - plot for sequence

function setup()

	local settings = experiment_settings()
	settings:setDefault()
	
	SAMPLE_RATE 					= 50-- Hz
	REF_REFRESH_INTERVAL 	= 3 -- s
	NR_JOINTS 						=	2
	LOOKAHEAD 						= 2.5 --s  
	EPISODES 							= 20
	EPISODE_LENGTH 				= 20 -- seconds
	EPISODE_SIZE 					= EPISODE_LENGTH * SAMPLE_RATE -- Datapoints
	TRIAL_SIZE						= EPISODE_SIZE * EPISODES
------------------ Communicator ----------------------------------------------
--[[ 			Used both for communicating with the other modules (i.e. training module, setup etc) and to keep (track of) time. --]]
	communicator 		= communicatorPB({sampling_time = 1/SAMPLE_RATE},true)
	channel 				= communicator:add_channel(
		{
			name 			= "gazebo_arm_plugin_comm", 
			bindpub 	= "tcp://*:5555", 
			bindsub		= "tcp://*:5556"
		})
	communicator:wait_for_synch()

------------------ Experience Memory -----------------------------------------
--[[ Used to collect the experience tuples and observations ]]--
	em_settings = {
		full_state_dimension 		= {torch.LongStorage({6})},
		--observation_dimension 	= nil, -- multimodal example: for a 50x50 rgbd image where the d is seperate: 	
		--{torch.LongStorage({50,50,3}) , torch.LongStorage({50,50}}		
		observation_dimension 	= {torch.LongStorage({3,3}), torch.LongStorage({5})},		  
		experience_replay_size 	= EPISODE_SIZE*10,
		overwrite_policy 				= 'HALF', -- 'FIFO',
		RL_state_parts 					= {
			observation = false, 
			state 			= true, 
			action 			= true, 
			next_state	= true, 
			reward 			= true, 
			next_action = false
		},
		action_dimension 	= {torch.LongStorage({2})},
		get_OSAR_function = function(self,time_index)
			return -- Empty function for the policy executor as 
		end,
		send_OSAR_function = function(self,time_index)
			return -- Empty function for the training function
		end,  
	}
	
	xpm 						= experience_memory(true,true,em_settings) 

---------------- Neural networks -------------------------------------------

	networks = drl_ddpg(opt.exec,opt.train, {
			statesize 		= 6,--TODO: automate
			actionsize 		=	2,
			actor					=
				{
					hsizes 				= {94,94},
					nonlinearity 	= nn.ReLu,
					batchnorm 		= true,
				}, 
			critic 				=
				{
					actionlayer   = 1,
					hsizes 				= {94,94},
					nonlinearity 	= nn.ReLu,
					batchnorm 		= true,
				},				
			GPU						= true, -- for the training networks
	})

----------------- Policy ----------------------------------------------------
	local explorationPolicy  = drl_policy_OU({theta = 0.6, sigma = 0.12, sample_rate = SAMPLE_RATE, action_dimension = NR_JOINTS})

	policy 													= drl_policy({action_dimension = NR_JOINTS,	
		 	communicator 								= communicator,
		 	action_dimension 						=	2,
		 	exploitationActionFunction 	= function (state) return networks:get_policy_action(state) end,
		 	explorationActionFunction		= explorationPolicy,		 
		 	tradeofftype								= 'add',
		 	bounds											= {-1,1},
		 	explorationAmountFunction		= drl_exploration_amount_function({
		 		functiontype 								= 'exponential_per_sequence',
		 		initial_exploration  				=	1.0,
		 		multiplier									= 0.85,
		 		minimum_exploration					= 0.1,
 			}), 
	}) 	

----------------- Experiment description ------------------------------------

	armangles = generalized_state({
		name = 'arm_angles', 
		experience_memory = xpm, 
		dimension = torch.LongStorage({NR_JOINTS}), 
		extrap = "zoh",
		statetype = "torch.DoubleTensor",
		prevPerFullState = 0,
		getFunction = function(self,time_index)
			return channel:get_state_part(time_index,'armstate','state')
		end
	})

	armanglevelocities = generalized_state({
		name = 'arm_angle_velocities', 
		experience_memory = xpm, 
		dimension = torch.LongStorage({NR_JOINTS}), 
		extrap = "zoh",
		statetype = "torch.DoubleTensor",
		prevPerFullState = 0,
		getFunction = function(self,time_index)
			return channel:get_state_part(time_index,'armstate','first_derivative')
		end
	})

	reference = generalized_state({
		name = 'reference', 
		experience_memory = xpm, 
		dimension = torch.LongStorage({NR_JOINTS}), 
		extrap = "zero",
		statetype = "torch.DoubleTensor",
		prevPerFullState = 0,
		ref = torch.Tensor(2):zero(),
		getFunction = function(self,time_index)
			if (time_index % SAMPLE_RATE*REF_REFRESH_INTERVAL == 0) then
				self.state_properties.ref:zero()
				local anglesum = 0
				for a = 1,NR_JOINTS do
					local angle = math.random()-0.5 -- [-0.5 rad , 0.5 rad]
					anglesum = anglesum + angle
					self.state_properties.ref[1] = self.state_properties.ref[1] + math.sin(anglesum) -- x
					self.state_properties.ref[2] = self.state_properties.ref[2] + math.cos(anglesum) -- y
				end
			end
			return self.state_properties.ref
		end
	})

--[[	
	test_observation1 = generalized_state({
		name = 'test_observation1', 
		experience_memory = xpm, 
		dimension = torch.LongStorage({3,3}), 
		extrap = "zoh",
		statetype = "torch.DoubleTensor",
		getFunction = function(self,time_index)
			return torch.Tensor(3,3):fill(time_index)
		end
	})
	
	test_observation2 = generalized_state({
		name = 'test_observation2', 
		experience_memory = xpm, 
		dimension = torch.LongStorage({5}), 
		extrap = "zoh",
		statetype = "torch.DoubleTensor",
		getFunction = function(self,time_index)
			return torch.Tensor(5):fill(time_index)
		end
	})

--]]
	action = generalized_state({
		name = 'action', 
		experience_memory = xpm, 
		dimension = torch.LongStorage({2}), 
		extrap = "zero",
		statetype = "torch.DoubleTensor",
		getFunction 	= function(self,time_index,full_state)
			local action 	= policy(full_state[1],time_index,communicator:get_sequence_index())
			channel:send_action("arm",action)
			return action
		end
	})
	
	cost = generalized_state({
		name = 'cost', 
		experience_memory = xpm, 
		dimension = torch.LongStorage({1}), 
		extrap = "nil",
		statetype = "torch.DoubleTensor",
		getFunction = function(self,time_index)
				local x,y,anglesum = 0,0,0
				local angles = armangles:get_value_for_timestep(time_index,communicator:get_sequence_index())
				for a = 1,angles:nElement() do
					anglesum = anglesum + angles[a]
					x = x + math.sin(anglesum)
					y = y + math.cos(anglesum) 
				end
				local distanceSQ = ((torch.Tensor({x,y}) - reference:get_value_for_timestep(time_index-1)):norm())^2	
				local speed = armanglevelocities:get_value_for_timestep(time_index):norm()
				local w1,w2,w3,alpha = 0.2,0.4,8,100
				return w1*distanceSQ + w2*(1-math.exp(-alpha*distanceSQ)) + w3*speed
		end
	})

	xpm:addPartialState(1,armangles)
	xpm:addPartialState(1,armanglevelocities)
	xpm:addPartialState(1,reference)
	xpm:setAction(action)
	xpm:setReward(cost,{delay=true,cost=true}) -- calculated reward is for the previous state action pair (based on s,a,s'), reward is infact a cost (minimize instead of maximize)
--	xpm:addObservation(test_observation1)
--	xpm:addObservation(test_observation2)
	
	diagnostics = drl_diagnostics({})
	diagnostics:addConsoleSummary(cost,{})
	diagnostics:addConsoleSummary(action,{})
	
end

function main()
	-- settings through command line
	local cmd = torch.CmdLine()
	-- note torch cant deal with passing booleans, therefore these options are false unless the program is called with their flags and no arguments.
	cmd:option('-noexec',false, 'This module does notinteract with the environment.')
	cmd:option('-notrain',false, 'This module does not train the networks.')
	opt = cmd:parse(arg)
	opt.exec 	= not(opt.noexec)
	opt.train = not(opt.notrain)
	
	setup()	
	
	communicator:reset_clock()
	-- main loop -----------------------
  for i = 1,EPISODES do
	 	for j = 1,SAMPLE_RATE * EPISODE_LENGTH  do
			local time_index 			= communicator:get_time_index()
			local sequence_index 	= communicator:get_sequence_index()
			if (opt.exec) then
					xpm:collect_OSAR(time_index,sequence_index) -- collect the (observations) state action and reward by interacting with the environment
				
			else -- no policy execution in this module, optionally receive (O)SAR tuples from another module
					xpm:receive_OSAR(time_index) -- functionality depends on the receivesar function that is given to the database.
			end

			if (opt.train) then
					xpm:add_RL_state_to_db()
			else -- no training in this module, optionally receive updated policy parameters from another module
				if (not(opt.noexec)) then -- if this module interacts with the environment but does not train then optionally send the SAR tuples to other module(s).
					xpm:send_OSAR(time_index)
				
				end		
			end
 		diagnostics:update(time_index, sequence_index)
		communicator:advance_clock()	-- advancing the clock increases the time index by one, sends it out on all channels and then blocks untill at least one sampling time period has passed since the last time the clock advancement function call was completed.

--		print(communicator:getLastLoad())
		end -- episode
 		networks:train(xpm,{
 			batch_size		= 64, 
 			batchupdates 	= 1*EPISODE_SIZE, 
 			gamma 				= math.exp(-((1/SAMPLE_RATE)/LOOKAHEAD)),
 			optimsettings = {
 				optimfunction = optim.adam,
 				configActor 	= {learningRate = 1e-3},
 				configCritic 	= {learningRate = 1e-4},
 				critic_L2norm	= 0.005,
 				targetlowpass = 1e-3
 			}
 		})
 		
 		networks:set_controller_parameters(networks:get_controller_parameters())
	  
 		communicator:advance_sequence_index()	
 		
	end -- training run		
	 		
xpm:saveToFile("test.db")
end  
 
main()
