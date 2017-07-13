--[[

		Tim de Bruin 2016
		deepRL-torch
		Fantasy experience class
		Uses a dynamics network / function and a reward function / network to create synthetic experiences. 
		Additionally, a classifier can be created that can determine the uniqueness of samples based on their components.
		The classifier can be used to train the dynamics model in a GAN fashion so that it can use noise inputs to model unobserved state variables.
		
--]]

require 'nn'
require 'nngraph'

local daydream = torch.class('drl_daydream')

--[[
experience_memory: drl_experience memory that is used to obtain the training samples 
settings{
	state_dimension
	action_dimension
	!optional: noise_dimension
	dynamics{
		!either:
			network{
				training 			= !either: 'DIRECT' !or 'GAN'
				hsizes 				= {l1,l2,...,ln}, 
				nonlinearity 	= nn.ReLu,
				batchnorm 		= boolean 	
				GPU						= boolean
			}
		!or:
			dynamicsfunction: next_state = function(state,action,noise)
	}
	reward{
		!either:
			network{
				training = !either: 'DIRECT' !or 'GAN'
			}
		!or:
			rewardfunction: reward = function(state, action, next_state, noise)
	}
	!optional:
	terminalstate{
		!either:
		network{
			training 			= !either: 'DIRECT' !or 'GAN'
			hsizes 				= {l1,l2,...,ln}, 
			nonlinearity 	= nn.ReLu,
			batchnorm 		= boolean 	
			GPU						= boolean 
		}
		!or:
		terminalfunction: 0 / 1 = function(next_state)
	}
	!optional: 
	classifier{
		training = !either: 'DIRECT' !or 'GAN'
	}
	database{
		trainfraction
		validationfraction
		testfraction
	}

}
--]]
function daydream:__init( experience_memory, settings )
	assert(settings, 'No settings given')
	assert(experience_memory, 'Experience_memory not specified')
	assert(settings.state_dimension,'State dimenision not set')
	print(settings.state_dimension)
	assert(settings.action_dimension,'Action dimension not set')
	self.settings = settings
	self.xpm = experience_memory
	assert(settings.database,"No database settings given")
	self.xpm:setsplitfractions( settings.database.trainfraction, settings.database.validationfraction, settings.database.testfraction )

	
	if self.settings.dynamics.network then
		self.dynamicsmodel =  self:makeDynamicsNetwork()
		self.predict_next_state = function (daydreammodule,state, action, noise) -- noise input is only relevant for dynamics models trained with the GAN method that have unobserved parts of the state space.
			if daydreammodule.settings.dynamics.noise_dimension and daydreammodule.settings.dynamics.noise_dimension[1] > 0 then
				return daydreammodule.dynamicsmodel.network:forward({state,action,noise})
			else
				return daydreammodule.dynamicsmodel.network:forward({state,action})
			end
		end		
	elseif self.settings.dynamics.dynamicsfunction  then
		self.predict_next_state = function (daydreammodule,state, action, noise)
			return daydreammodule.settings.dynamics.dynamicsfunction(state,action,noise)
		end
	else
		print("No state dynamics predictor / function specified!")
	end
	if self.settings.reward.network then
		self.rewardmodel = self:makeRewardNetwork()
		self.predict_reward = function (daydreammodule,state, action, next_state)
			return daydreammodule.rewardmodel.network:forward({state,action,next_state})
		end
	elseif self.settings.reward.rewardfunction  then
		self.predict_reward = function (daydreammodule, state, action, next_state)
			local reward
				if #state:size() == 2 then
				reward = torch.Tensor(state:size(1))
			else
				reward = torch.Tensor(1)
			end
			for bi = 1,reward:nElement() do
				reward[bi] = daydreammodule.settings.reward.rewardfunction(state[bi]:double(),action[bi]:double(),next_state[bi]:double())
			end
			reward:typeAs(state)
			return reward
		end
	else
		print("No state reward predictor / function specified!")
	end
	if self.settings.terminalstate then
		if self.settings.terminalstate.network then
			self.terminalmodel = self:makeTerminalNetwork()
			self.predict_terminal = function (daydreammodule,next_state)
				return daydreammodule.terminalmodel.network:forward(next_state)
			end
		elseif self.settings.terminalstate.terminalfunction  then
			self.predict_terminal = function (daydreammodule,next_state)
			local terminal
				if #next_state:size() == 2 then
				terminal = torch.Tensor(next_state:size(1))
			else
				terminal = torch.Tensor(1)
			end
			for bi = 1,terminal:nElement() do
				terminal[bi] = daydreammodule.settings.terminalstate.terminalfunction(next_state[bi]:double())
			end
			terminal:typeAs(next_state)
			return terminal
			end
		else
			print("No state terminal predictor / function specified!")
		end
	end
	if self.settings.state then
		if self.settings.state.network then
			self.statemodel = self:makeStateNetworks()
			self.sample_state = function (daydreammodule,batch_size)
				local noise_input = torch.Tensor(batch_size,daydreammodule.statemodel.generator.noise_dimension):uniform(-1,1)
				if settings.GPU then noise_input:cuda() end
				return daydreammodule.statemodel.generator.network:forward(noise_input)
			end
		end
	end
end

function daydream:makeDynamicsNetwork()
	assert(self.settings.dynamics.network,'dynamicsnetwork properties missing')
	local settings = self.settings.dynamics.network
	local nonlinearity			= settings.nonlinearity or nn.ReLU
	local batchnorm					= nn.Identity
	if settings.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
  local stateInput 				=	nn.Identity()()
	local actionInput 			= nn.Identity()()	
	local noiseInput 				= nn.Identity()()
	local hiddenlayers 			= {}
	local layersizes				= {}
	local usenoise 					= (self.settings.dynamics.noise_dimension and self.settings.dynamics.noise_dimension[1]> 0)
	assert(#self.settings.state_dimension == 1 and #self.settings.action_dimension == 1, "State and action dimensions in R1 expected. (Conv predictions not yet implemeneted)")
	if usenoise then
		hiddenlayers[0] = nn.JoinTable(3)({stateInput, actionInput, noiseInput})
		layersizes[0] 	= self.settings.state_dimension[1] + self.settings.action_dimension[1] + self.settings.noise_dimension[1]
	else
		hiddenlayers[0] = nn.JoinTable(2)({stateInput, actionInput})
		layersizes[0] 	= self.settings.state_dimension[1] + self.settings.action_dimension[1]
	end
	for layerindex = 1,#settings.hiddensizes do	
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],settings.hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= settings.hiddensizes[layerindex]
	end 
	local next_state_output			= nn.Linear(layersizes[#settings.hiddensizes],self.settings.state_dimension[1])(hiddenlayers[#hiddenlayers])
	local network
	if usenoise then
 		network 									= nn.gModule({stateInput, actionInput, noiseInput},{next_state_output})
	else
		network 									= nn.gModule({stateInput, actionInput},{next_state_output})
	end
	if settings.GPU then 
		network:cuda()
	end
	local paramx, paramdx = network:getParameters()
	local dynamics_table = { 
		network = network,
		paramx  = paramx,
		paramdx = paramdx, 
	}
	return dynamics_table
end

function daydream:makeRewardNetwork()
	assert(self.settings.reward.network,'rewardnetwork properties missing')
	local settings = self.settings.reward.network
	local nonlinearity			= settings.nonlinearity or nn.ReLU
	local batchnorm					= nn.Identity
	if settings.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
  local stateInput 				=	nn.Identity()()
	local actionInput 			= nn.Identity()()	
	local next_stateInput		= nn.Identity()()
	local hiddenlayers 			= {}
	local layersizes				= {}
	assert(#self.settings.state_dimension == 1 and #self.settings.action_dimension == 1, "State and action dimensions in R1 expected. (Conv predictions not yet implemeneted)")
	hiddenlayers[0] = nn.JoinTable(2)({stateInput, actionInput, next_stateInput})
	layersizes[0] 	= self.settings.state_dimension[1] + self.settings.action_dimension[1] + self.settings.state_dimension[1]
	
	for layerindex = 1,#settings.hiddensizes do	
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],settings.hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= settings.hiddensizes[layerindex]
	end 
	local reward_output					= nn.Linear(layersizes[#settings.hiddensizes],1)(hiddenlayers[#hiddenlayers])
	local network 							= nn.gModule({stateInput, actionInput, next_stateInput},{reward_output})
	if settings.GPU then 
		network:cuda()
	end
	local paramx, paramdx = network:getParameters()
	local reward_table = { 
		network = network,
		paramx  = paramx,
		paramdx = paramdx, 
	}
	return reward_table
end

function daydream:makeTerminalNetwork()
	assert(self.settings.terminalstate.network,'terminalnetwork properties missing')
	local settings 					= self.settings.terminalstate.network
	local nonlinearity			= settings.nonlinearity or nn.ReLU
	local batchnorm					= nn.Identity
	if settings.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
	local next_stateInput		= nn.Identity()()
	local hiddenlayers 			= {}
	local layersizes				= {}
	hiddenlayers[0] = next_stateInput
	layersizes[0] 	= self.settings.state_dimension[1]
	for layerindex = 1,#settings.hiddensizes	do
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],settings.hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= settings.hiddensizes[#hiddenlayers]
	end 
	local terminal_output				= nn.Sigmoid()(nn.Linear(layersizes[#settings.hiddensizes],1)(hiddenlayers[#hiddenlayers]))
	local network 							= nn.gModule({next_stateInput},{terminal_output})
	if settings.GPU then 
		network:cuda()
	end
	local paramx, paramdx = network:getParameters()
	local terminal_table = { 
		network = network,
		paramx  = paramx,
		paramdx = paramdx, 
	}
	return terminal_table
end

function daydream:makeStateNetworks()
	assert(self.settings.state.network,'terminalnetwork properties missing')
	local settings = self.settings.state.network
	local nonlinearity			= settings.generator.nonlinearity or nn.ReLU
	local batchnorm					= nn.Identity
	if settings.generator.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
	local noise_input				= nn.Identity()()
	local hiddenlayers 			= {}
	local layersizes				= {}
	hiddenlayers[0] = noise_input
	layersizes[0] 	= settings.generator.noise_dimension
	for layerindex = 1,#settings.generator.hiddensizes	do
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],settings.generator.hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= settings.generator.hiddensizes[#hiddenlayers]
	end 
	local state_output					= nn.Linear(layersizes[#settings.generator.hiddensizes],self.settings.state_dimension[1])(hiddenlayers[#hiddenlayers])
	local network 							= nn.gModule({noise_input},{state_output})
	if settings.generator.GPU then 
		network:cuda()
	end
	local paramx, paramdx = network:getParameters()
	local state_table = { 
		generator = {},
		discriminator = {},
	 } 
	state_table.generator.noise_dimension = settings.generator.noise_dimension
	state_table.generator.paramx  = paramx
	state_table.generator.paramdx = paramdx
	state_table.generator.network = network
	
	-- discriminator
	local nonlinearity			= settings.discriminator.nonlinearity or nn.ReLU
	local batchnorm					= nn.Identity
	if settings.discriminator.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
	local stateInput				= nn.Identity()()
	local hiddenlayers 			= {}
	local layersizes				= {}
	hiddenlayers[0] = stateInput
	layersizes[0] 	= self.settings.state_dimension[1]
	for layerindex = 1,#settings.discriminator.hiddensizes	do
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],settings.discriminator.hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= settings.discriminator.hiddensizes[#hiddenlayers]
	end 
	local classification_output			= nn.Sigmoid()(nn.Linear(layersizes[#settings.discriminator.hiddensizes],1)(hiddenlayers[#hiddenlayers]))
	local network 									= nn.gModule({stateInput},{classification_output})
	if settings.discriminator.GPU then 
		network:cuda()
	end
	local paramx, paramdx = network:getParameters()
	state_table.discriminator.network = network
	state_table.discriminator.paramx  = paramx
	state_table.discriminator.paramdx = paramdx
	state_table.discriminator.update_count = 0

	return state_table
end


function daydream:train_networks(experiencedb, trainsettings)
	self.tempTrainSettings = trainsettings
	self.MSECriterion = nn.MSECriterion() 
	self.BCECriterion = nn.BCECriterion() 
	if trainsettings.GPU then
		self.MSECriterion:cuda()
		self.BCECriterion:cuda()
	end

	-- determine which networks to train
	-- while still networks to train 
		-- get a train batch
		-- for each network to train
			-- train update with batch
		-- get a validation batch
		-- for each network to train
			-- check convergence, update trainme
		-- update DB stats		

	local function vprint(string)
		if trainsettings.verbal then
			print(string)
		end
	end

	local function evaluateDynamicsDIRECT(paramx_)
		if paramx_ ~= self.dynamicsmodel.paramx then self.dynamicsmodel.paramx:copy(paramx_) end
		self.dynamicsmodel.paramdx:zero()
		local predicted_next_state = self.dynamicsmodel.network:forward({self.state,self.action})
		local loss = self.MSECriterion:forward(predicted_next_state,self.next_state)
		self.dynamicsmodel.network:backward({self.state,self.action},self.MSECriterion:backward(predicted_next_state,self.next_state))
		return 0, self.dynamicsmodel.paramdx	
	end

	local function evaluateRewardDIRECT(paramx_)
		if paramx_ ~= self.rewardmodel.paramx then self.rewardmodel.paramx:copy(paramx_) end
		self.rewardmodel.paramdx:zero()
		local predicted_reward = self.rewardmodel.network:forward({self.state,self.action,self.next_state})
		local loss = self.MSECriterion:forward(predicted_reward,self.reward)
		self.rewardmodel.network:backward({self.state,self.action,self.next_state},self.MSECriterion:backward(predicted_reward,self.reward))
		return 0, self.rewardmodel.paramdx	
	end

	local function evaluateTerminalDIRECT(paramx_)
		if paramx_ ~= self.terminalmodel.paramx then self.terminalmodel.paramx:copy(paramx_) end
		self.terminalmodel.paramdx:zero()
		local predicted_terminal = self.terminalmodel.network:forward(self.next_state)
		local loss = self.BCECriterion:forward(predicted_terminal,self.terminal)
		self.terminalmodel.network:backward({self.next_state},self.BCECriterion:backward(predicted_terminal,self.terminal))
		return 0, self.terminalmodel.paramdx	
	end	

	local function evaluateStateDiscriminatorGAN(paramx_)
		if paramx_ ~= self.statemodel.discriminator.paramx then self.statemodel.discriminator.paramx:copy(paramx_) end
		self.statemodel.discriminator.paramdx:zero()
		-- start with a batch of real samples
		self.statemodel.target:fill(1)
		local classifications = self.statemodel.discriminator.network:forward(self.state)
		local real_loss = self.BCECriterion:forward(classifications,self.statemodel.target)
		self.statemodel.discriminator.network:backward(self.state,self.BCECriterion:backward(classifications,self.statemodel.target))
		-- then a batch of fantasies
		self.statemodel.target:fill(0)
		self.statemodel.last_noise:uniform(-1,1)
		self.statemodel.last_state = self.statemodel.generator.network:forward(self.statemodel.last_noise)
		classifications = self.statemodel.discriminator.network:forward(self.statemodel.last_state)
		self.statemodel.discriminator.last_classifications = classifications
		local fantasy_loss = self.BCECriterion:forward(classifications,self.statemodel.target)
		self.statemodel.discriminator.network:backward(self.statemodel.last_state,self.BCECriterion:backward(classifications,self.statemodel.target))
		return (real_loss+fantasy_loss), self.statemodel.discriminator.paramdx	
	end	

	local function evaluateStateGeneratorGAN(paramx_)
		if paramx_ ~= self.statemodel.generator.paramx then self.statemodel.generator.paramx:copy(paramx_) end
		self.statemodel.generator.paramdx:zero()
		self.statemodel.target:fill(1) -- loss is opposite of the discriminator
		local GenLoss 	= self.BCECriterion:forward(self.statemodel.discriminator.last_classifications,self.statemodel.target)
		local dDiscrim 	= self.BCECriterion:backward(self.statemodel.discriminator.last_classifications,self.statemodel.target)
   	local dFantasy 	= self.statemodel.discriminator.network:updateGradInput(input, dDiscrim) -- dont accumulate gradients
   	self.statemodel.generator.network:backward(self.statemodel.last_noise,dFantasy)
		return GenLoss, self.statemodel.generator.paramdx	
	end	


-- determine which networks to train
	local networks_to_train = {
		dynamics 	= self.dynamicsmodel 	~= nil,
		reward 		= self.rewardmodel 		~= nil,
		terminal 	= self.terminalmodel 	~= nil,
		state 		= self.statemodel 		~= nil,
		action 		= self.actionmodel 		~= nil,
	}
	networks_to_train.any = function (self)
		return self.dynamics or self.reward or self.terminal or self.state or self.action 
	end	
	if networks_to_train.state and (not(self.statemodel.last_noise) or self.statemodel.last_noise:size(1) ~= trainsettings.batch_size) then
		self.statemodel.last_noise = torch.Tensor(trainsettings.batch_size,self.statemodel.generator.noise_dimension)
		self.statemodel.target = torch.Tensor(trainsettings.batch_size,1)
		self.statemodel.last_state = torch.Tensor(trainsettings.batch_size,self.settings.state_dimension[1])
		if trainsettings.GPU then
			self.statemodel.last_noise:cuda()
			self.statemodel.last_state:cuda()
			self.statemodel.target:cuda()
		end
	end

	-- while there are still networks to train 
	local epoch_counter = 0
	while networks_to_train:any() and epoch_counter < trainsettings.max_epochs do
		epoch_counter = epoch_counter + 1
		-- get the train batches
		local traindata			= experiencedb:get_mini_batch(trainsettings.batch_size, trainsettings.GPU, nil, nil, false, "train")
		-- perform an epoch of updates to all the networks to be trained
		for minibatch_counter = 1,traindata.db_indices:size(1) do
			local statepart = 1
			local timeseqidx = 1
			self.state = traindata.state[statepart][minibatch_counter][timeseqidx]
			self.action = traindata.action[statepart][minibatch_counter][timeseqidx]
			self.next_state = traindata.next_state[statepart][minibatch_counter][timeseqidx]
			self.reward = traindata.reward[minibatch_counter][timeseqidx]
			if traindata.terminal then
				self.terminal = traindata.terminal[minibatch_counter][timeseqidx]:clone()
			else
				self.terminal = self.reward:clone():zero()
			end			

			if networks_to_train.dynamics then
				if self.settings.dynamics.network.training == 'DIRECT' then
					trainsettings.dynamics.optimfunction(evaluateDynamicsDIRECT, self.dynamicsmodel.paramx, trainsettings.dynamics.optim_settings , trainsettings.dynamics.optim_state)
				else
					assert(false, "not implemeneted")
				end
			end
			if networks_to_train.reward then
				if self.settings.reward.network.training == 'DIRECT' then
					trainsettings.reward.optimfunction(evaluateRewardDIRECT, self.rewardmodel.paramx, trainsettings.reward.optim_settings , trainsettings.reward.optim_state)
				else
					assert(false, "not implemeneted")
				end
			end
			if networks_to_train.terminal then
				if self.settings.terminalstate.network.training == 'DIRECT' then
					trainsettings.terminalstate.optimfunction(evaluateTerminalDIRECT, self.terminalmodel.paramx, trainsettings.terminalstate.optim_settings , trainsettings.terminalstate.optim_state)
				else
					assert(false, "not implemeneted")
				end
			end
			if networks_to_train.state then
				if self.settings.state.network.training == 'GAN' then
					self.statemodel.last_noise:uniform(-1,1)
					trainsettings.state.discriminator.optimfunction(evaluateStateDiscriminatorGAN, self.statemodel.discriminator.paramx, trainsettings.state.discriminator.optim_settings , trainsettings.state.discriminator.optim_state)
					self.statemodel.discriminator.update_count = self.statemodel.discriminator.update_count + 1
					if self.statemodel.discriminator.update_count % trainsettings.state.dg_ratio == 0 then 
						trainsettings.state.generator.optimfunction(evaluateStateGeneratorGAN, self.statemodel.generator.paramx, trainsettings.state.generator.optim_settings , trainsettings.state.generator.optim_state)
					end
				else
					assert(false, "not implemeneted")
				end
			end
		end -- end of training epoch
		-- get the validation data
		local valdata			= experiencedb:get_mini_batch(trainsettings.batch_size, trainsettings.GPU, nil, nil, false, "validate")
		local lossD, lossDS, lossR, lossT, lossS, lossA = 0,0,0,0,0,0
		-- perform an epoch of loss checks to all the networks still being updated
		for minibatch_counter = 1,valdata.db_indices:size(1) do
			local statepart = 1
			local timeseqidx = 1
			self.state = valdata.state[statepart][minibatch_counter][timeseqidx]
			self.action = valdata.action[statepart][minibatch_counter][timeseqidx]
			self.next_state = valdata.next_state[statepart][minibatch_counter][timeseqidx]
			self.reward = valdata.reward[minibatch_counter][timeseqidx]
			if valdata.terminal then
				self.terminal = valdata.terminal[minibatch_counter][timeseqidx]:clone()
			else
				self.terminal = self.reward:clone():zero()
			end			

			if networks_to_train.dynamics then
				lossD = lossD + self.MSECriterion:forward(self.dynamicsmodel.network:forward({self.state,self.action}),self.next_state)	
				lossDS = lossDS + self.MSECriterion:forward(self.state, self.next_state)
			end
			if networks_to_train.reward then
				lossR = lossR + self.MSECriterion:forward(self.rewardmodel.network:forward({self.state,self.action,self.next_state}),self.reward)
			end
			if networks_to_train.terminal then
				lossT = lossT + self.BCECriterion:forward(self.terminalmodel.network:forward(self.next_state),self.terminal)
			end
			if networks_to_train.state then
				lossS = lossS + self.statemodel.discriminator.network:forward(self.state):mean()	
			end
			if networks_to_train.action then
				assert(false, "Check not implemeneted yet")
			end
		end -- end of validation epoch
		-- for each network left to train
		-- check convergence, update trainme	
		vprint("Daydream networks train epoch " .. epoch_counter)
		if networks_to_train.dynamics then
				lossD = lossD / valdata.db_indices:size(1)
				lossDS = lossDS / valdata.db_indices:size(1)
				vprint("Dynamics model: " .. lossD .. " ( last: " .. trainsettings.dynamics.lastValLoss .. " , Static (sanity check: ".. lossDS .." )) ")
				if lossD < trainsettings.dynamics.lastValLoss then
					trainsettings.dynamics.lastValLoss = lossD
				else
					trainsettings.dynamics.lastValLoss = math.huge
					networks_to_train.dynamics = false
				end
		end
		if networks_to_train.reward then
				lossR = lossR / valdata.db_indices:size(1)
				vprint("Reward model: " .. lossR .. " ( last: " .. trainsettings.reward.lastValLoss .. " ) ")
				if lossR < trainsettings.reward.lastValLoss and lossR > 1e-6 then
					trainsettings.reward.lastValLoss = lossR
				else
					trainsettings.reward.lastValLoss = math.huge
					networks_to_train.reward = false
				end
		end
		if networks_to_train.terminal then
				lossT = lossT / valdata.db_indices:size(1)
				vprint("Terminal state model: " .. lossT .. " ( last: " .. trainsettings.terminalstate.lastValLoss .. " ) ")
				if lossT < trainsettings.terminalstate.lastValLoss and lossT > 1e-9 then
					trainsettings.terminalstate.lastValLoss = lossT
				else
					trainsettings.terminalstate.lastValLoss = math.huge
					networks_to_train.terminal = false
				end
		end
		if networks_to_train.state then
				lossS = lossS / valdata.db_indices:size(1)
				vprint("State generation model, validation mean likelihood: " .. lossS .. " ( last: " .. trainsettings.state.lastValLoss .. " ) ")
				if lossS < 0.5 or epoch_counter >=  trainsettings.state.max_epochs then
					networks_to_train.state = false
					if epoch_counter < 10 then
						trainsettings.state.dg_ratio = trainsettings.state.dg_ratio + 1
						vprint("Increasing dg_ratio to " .. trainsettings.state.dg_ratio )
					elseif epoch_counter >= trainsettings.state.max_epochs then
						trainsettings.state.dg_ratio = math.max(trainsettings.state.dg_ratio - 1,1)
						vprint("Decreasing dg_ratio to " .. trainsettings.state.dg_ratio )
					end
				end
				trainsettings.state.lastValLoss = lossS	
		end
		vprint(" ")
	end -- while there are networks left to train
	-- update DB stats
	if (trainsettings.dynamics and (trainsettings.dynamics.logDYNERROR or trainsettings.dynamics.logRELDYNERROR)) or (trainsettings.reward and trainsettings.reward.logREWERROR) or (trainsettings.state and trainsettings.state.discriminator.logSTATEUNLIKELINESS) then 
		vprint("Updating DB stats...") 
		local suprise_stats = {}
		local wholeDB			= experiencedb:get_mini_batch(50, trainsettings.GPU, nil, nil, false, nil) -- batchsize 100 for speed.
		-- perform an epoch of loss checks to all the networks still being updated
		for minibatch_counter = 1,wholeDB.db_indices:size(1) do
			local statepart 	= 1
			local timeseqidx 	= 1
			local db_indices	= wholeDB.db_indices[minibatch_counter][timeseqidx]
			local state 			= wholeDB.state[statepart][minibatch_counter][timeseqidx]
			local action 			= wholeDB.action[statepart][minibatch_counter][timeseqidx]
			local next_state 	= wholeDB.next_state[statepart][minibatch_counter][timeseqidx]
			local reward 			= wholeDB.reward[minibatch_counter][timeseqidx]
			
			if (trainsettings.dynamics and (trainsettings.dynamics.logDYNERROR or trainsettings.dynamics.logRELDYNERROR)) then
				local predicted_next_state = self.dynamicsmodel.network:forward({state,action})
				local state_error 				 = predicted_next_state:add(-1.0,next_state):pow(2):sum(2)
				if (trainsettings.dynamics.logDYNERROR) then
					experiencedb:update_extra_info("DYNERROR",state_error:clone():double(),db_indices)	
					suprise_stats.DYNERROR =  true
				end
				if (trainsettings.dynamics.logRELDYNERROR) then
					local EPSILON = 1e-6 -- to prevent div by zero
					local staticError = state:clone():add(-1.0,next_state):pow(2):sum(2):add(EPSILON)
					local relError = state_error:cdiv(staticError)
					experiencedb:update_extra_info("RELDYNERROR",relError:clone():double(),db_indices)	 
					suprise_stats.RELDYNERROR =  true
				end
			end
			
			if (trainsettings.reward and trainsettings.reward.logREWERROR) then
				local predicted_reward 	= self:predict_reward(state,action,next_state)
				local reward_error 			= predicted_reward:add(-1.0,reward):pow(2):sum(2)
				experiencedb:update_extra_info("REWERROR",reward_error:clone():double(),db_indices)	
				suprise_stats.REWERROR =  true
			end
			if (trainsettings.state and trainsettings.state.discriminator.logSTATEUNLIKELINESS) then
				local unlikeliness 	= self.statemodel.discriminator.network:forward(state)
				experiencedb:update_extra_info("STATEUNLIKELINESS",unlikeliness:clone():double(),db_indices)	
				suprise_stats.STATEUNLIKELINESS =  true
			end
		end
		self.performance_stats = experiencedb:update_surprise_flags(suprise_stats,trainsettings.verbal)

	end
			
end



-----------------------------------------------------------
-- Let the daydream module act as an experience database --
-----------------------------------------------------------

--[[
call this function to set the properties of the batches of synthetic experiences to be created, before passing the daydream object as an experience database
dreamsettings {
	real_database = drl_experience_database	
	epoch_size = .. (number of batches in an epoch)
	state_source = 'REAL' / 'DREAM' (whether to use states from the real databas or ones from a generative model)
	action_type = 'ONEHOT' / 'CONTINUOUS' / 'DISCRETE' -- the type of actions
	action_min = torch.Tensor(actiondim) <needed when action_type is cont or discrete>
	action_max = torch.Tensor(actiondim) <needed when action_type is cont or discrete>
}
]]
function daydream:setDreamParameters( dreamsettings, value )
	assert(dreamsettings, "dreamsettings nil")
	if value then
		self.experience_database.dreamSettings[dreamsettings] = value
	else
		self.experience_database = {} -- to trick the NN class into thinking this is an experience database
		self.experience_database.dreamSettings = dreamsettings
	end
end

-- Note: discrete actions are given in a onehot form.
function daydream:getRandomAction(batch_size)
	batch_size = batch_size or 1 
	local action
	if self.experience_database.dreamSettings.action_type == "CONTINUOUS" then
		if not(self.experience_database.action_sub) or self.experience_database.action_sub:size(1) ~= batch_size then
			self.experience_database.action_cadd = torch.Tensor(batch_size,self.settings.action_dimension[1])
			self.experience_database.action_cmul = torch.Tensor(batch_size,self.settings.action_dimension[1])
			for d = 1,self.settings.action_dimension[1] do
				self.experience_database.action_cadd[{{},d}]:fill(self.experience_database.dreamSettings.action_bounds.min[d])
				self.experience_database.action_cmul[{{},d}]:fill(self.experience_database.dreamSettings.action_bounds.max[d] - self.experience_database.dreamSettings.action_bounds.min[d])
			end
		end

		action = torch.rand(batch_size,self.settings.action_dimension[1]):cmul(self.experience_database.action_cmul):add(self.experience_database.action_cadd)
	else -- assume onehot
		action = torch.Tensor(batch_size,self.settings.action_dimension[1]):zero()
		for b = 1, batch_size do
			actionR = torch.random(self.settings.action_dimension[1])
			action[b][actionR] = 1
		end
	end
	return action
end

function daydream:get_mini_batch(batch_size,GPU,requested_parts)
	local requestedTensorType = torch.getdefaulttensortype()
	if GPU then
		requestedTensorType = "torch.CudaTensor"
	end
	local realprob
	if (self.experience_database.dreamSettings.reality_mix == 'FANTASY') then
		realprob = 0
	elseif(self.experience_database.dreamSettings.reality_mix == 'PROPORTIONAL') then 
		realprob = self.performance_stats.problemfraction or 1
	elseif(self.experience_database.dreamSettings.reality_mix == 'FULLSURPRISE') then
		realprob = 1
	elseif type(self.experience_database.dreamSettings.reality_mix)=='number' then 
		realprob = self.experience_database.dreamSettings.reality_mix
	else
		assert(false,"reality_mix unspecified or unknown")
	end

	local batch = {}
	local reqp = requested_parts or {observation = false, state = true, action = true, next_state = true, reward = true, terminal = true}
	assert(not(reqp.observation),"Synthetic observations not (yet) implemeneted")
	
	local statesize = self.settings.state_dimension:totable()
	table.insert(statesize,1,batch_size)
	table.insert(statesize,1,1)
	table.insert(statesize,1,self.experience_database.dreamSettings.epoch_size)	
	local actionsize = {self.experience_database.dreamSettings.epoch_size,1,batch_size,self.settings.action_dimension[1]}
	local rewardsize = {self.experience_database.dreamSettings.epoch_size,1,batch_size,1}	
	local terminalsize = {self.experience_database.dreamSettings.epoch_size,1,batch_size,1}	

	if GPU then 
		batch.state 			= {torch.CudaTensor(unpack(statesize))} 
		batch.next_state 	= {torch.CudaTensor(unpack(statesize))} 
		batch.action 			= {torch.CudaTensor(unpack(actionsize))}
		batch.reward 			= torch.CudaTensor(unpack(rewardsize))
		batch.terminal 		= torch.CudaTensor(unpack(terminalsize))
	else 
		batch.state 			= {torch.Tensor(unpack(statesize))}
		batch.next_state 	= {torch.Tensor(unpack(statesize))}
		batch.action 			= {torch.Tensor(unpack(actionsize))}
		batch.reward 			= torch.Tensor(unpack(rewardsize)) 
		batch.terminal 		= torch.Tensor(unpack(terminalsize)) 
	end


	for bi=1,statesize[1] do --for each batch in the fake DB get the components	
		--state
		
		if math.random() < self.experience_database.dreamSettings.unireality_prob then
			local realbatch = self.experience_database.dreamSettings.real_database:get_mini_batch(batch_size,GPU,nil,nil,true) -- normal batch
			batch.state[1][bi]:copy(realbatch.state[1])
			batch.action[1][bi]:copy(realbatch.action[1])
			batch.next_state[1][bi]:copy(realbatch.next_state[1])
			batch.reward[bi]:copy(realbatch.reward)
			if reqp.terminal then
				batch.terminal[bi]:copy(realbatch.terminal)
			end
		else
			if math.random() < realprob then -- surpise batch
				local realbatch = self.experience_database.dreamSettings.real_database:get_mini_batch(batch_size,GPU,nil,nil,true,nil,true)
				batch.state[1][bi]:copy(realbatch.state[1])
				batch.action[1][bi]:copy(realbatch.action[1])
				batch.next_state[1][bi]:copy(realbatch.next_state[1])
				batch.reward[bi]:copy(realbatch.reward)
				if reqp.terminal then
					batch.terminal[bi]:copy(realbatch.terminal)
				end

			else -- fanasy batch
				if self.experience_database.dreamSettings.state_source == 'REAL' then
					batch.state[1][bi]:copy((self.experience_database.dreamSettings.real_database:get_mini_batch(batch_size,GPU,{state=true},nil,true)).state[1])	
				elseif self.experience_database.dreamSettings.state_source == 'FANTASY' then 
					batch.state[1][bi]:copy(self:sample_state(batch_size))
				elseif self.experience_database.dreamSettings.state_source == 'ROLLOUT' then 
					assert(self.experience_database.dreamSettings.rollout and self.experience_database.dreamSettings.rollout > 0)
					if (bi==1 or bi%self.experience_database.dreamSettings.rollout == 0) then
						--print(self.experience_database.dreamSettings.rollout)
						batch.state[1][bi]:copy((self.experience_database.dreamSettings.real_database:get_mini_batch(batch_size,GPU,{state=true},nil,true)).state[1])	
					else
						local replacements = self.experience_database.dreamSettings.real_database:get_mini_batch(batch_size,GPU,{state=true},nil,true).state[1]
						batch.state[1][bi]:copy(batch.next_state[1][bi-1])
						for b = 1,batch_size do
							if batch.terminal[bi-1][1][b][1] > 0.9 then 
								batch.state[1][bi][1][b]:copy(replacements[1][1][b])
							end
						end
					end
				else
					assert(false,"state_source should be either 'REAL' or 'FANTASY' or 'ROLLOUT'.")
				end
				-- action
				batch.action[1][bi][1]:copy(self:getRandomAction(batch_size):type(requestedTensorType))
				-- next state
				-- TODO: NOISE!
				if self.settings.dynamics.noise_dimension and self.settings.dynamics.noise_dimension[1]> 0 then
					assert(false,"quickly implement this bit")
				else
					batch.next_state[1][bi][1]:copy(self:predict_next_state(batch.state[1][bi][1],batch.action[1][bi][1]))
				end	
				-- reward
				batch.reward[bi][1]:copy(self:predict_reward(batch.state[1][bi][1],batch.action[1][bi][1],batch.next_state[1][bi][1]))
				-- terminal state
				if reqp.terminal then
					batch.terminal[bi][1]:copy(self:predict_terminal(batch.next_state[1][bi][1]))
				end
			end
		end
	end
	batch.db_indices = batch.terminal:clone():zero()
	return batch
end

function daydream:update_extra_info(name,values,indices)
	-- do something with it?
end


