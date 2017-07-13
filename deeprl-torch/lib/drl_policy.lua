--[[
		Tim de Bruin 2015
		deepRL-torch
		policy class.
		
--]]

local pol = torch.class('drl_policy')

function pol:__init(settings)
	self.settings 						= settings
	self.bounds               = settings.bounds
  self.action 							= torch.Tensor(settings.action_dimension):zero()
	self.action_exploration 	= torch.Tensor(settings.action_dimension):zero()
	self.action_exploitation	= torch.Tensor(settings.action_dimension):zero()
	self.exploitationAction 	= settings.exploitationActionFunction
	self.explorationAction		= settings.explorationActionFunction
	self.explorationAmount  	= settings.explorationAmountFunction 
  self.lastActionWasExploritory = false
  if self.bounds then
  	self.mean                 = (self.bounds[2] + self.bounds[1]) /2 
  else
  	self.mean 								= 0
  end
end

function pol:__call(state,time_index,sequence_index)
	assert(state,"No state given")
	assert(time_index,"time_index is nil")
	assert(sequence_index,"sequence_index is nil")
	local exploration 						= self.explorationAmount(time_index,sequence_index)
	if (exploration > 0) then
		self.explorationAction:get_for_index(self.action_exploration,time_index)
	else
		self.action_exploration:zero()
	end
	if (exploration < 1) then
		self.action_exploitation = self.exploitationAction(state)
	else
		self.action_exploitation:fill(self.mean)
	end
	
	assert(self.settings.tradeofftype,"No tradeoff type set")
	if self.settings.tradeofftype=='add' then
		self.action = self.action_exploitation + (self.action_exploration:csub(self.mean))*exploration
	elseif self.settings.tradeofftype=='addunscaled' then
			self.action = self.action_exploitation + self.action_exploration
	elseif self.settings.tradeofftype=='scale' then
		self.action = (1-exploration)*self.action_exploitation + exploration*self.action_exploration
--[[		print("exploration amount:")
		print(exploration)
		print("exploration")
		print(self.action_exploration)
		print("policy:")
		print(self.action_exploitation)
		print("taken:")
		print(self.action)]]
	elseif self.settings.tradeofftype=='greedy' then
		if (math.random() < exploration) then
			self.action = self.action_exploration
			self.lastActionWasExploritory = true
		else
			self.action = self.action_exploitation
			self.lastActionWasExploritory = false
		end
	else
		assert(false,"unknown tradeoff type")
	end	
	if (self.settings.bounds) then
		self.action:clamp(self.settings.bounds[1],self.settings.bounds[2])
	end
	self.explorationeffect = self.action:clone():csub(self.action_exploitation)
	return self.action
end

function pol:get_exploration_effect(type)
	assert(self.explorationeffect,"explorationeffect not saved")
	return self.explorationeffect:norm(1)/self.explorationeffect:nElement()
end

function pol:deterministic(state)
    return self.exploitationAction(state)
end
	
function pol:getLastActionComponents()
	return {action = self.action, action_explore = self.action_exploration, action_pol = self.action_exploitation}
end

-- Ornstein-Uhlenbeck process (motion of a Brownian particle with friction) ---
local pol_OU = torch.class('drl_policy_OU')

function pol_OU:__init(settings)
	assert(settings.theta,"theta not set")
	assert(settings.sigma,"sigma not set")
	assert(settings.sample_rate,"sample_rate not set")
	assert(settings.action_dimension,"action_dimension not set")
	
	self.theta 						= settings.theta
	self.sigma 						= settings.sigma
	self.sample_rate			= settings.sample_rate
	self.lastAction 			= torch.Tensor(settings.action_dimension):zero() 
	self.mean 						= settings.mean or torch.Tensor(settings.action_dimension):zero()
	if settings.bounds then
		self.bounds = settings.bounds
	end


end

function pol_OU:get_for_index(action,time_index)
	local resolution = 1000
	if (math.max(1,math.floor(resolution/self.sample_rate)) > 100 and not(self.warned_about_ineff)) then
		print("WARNING: function drl_policy_OU:get_for_index is inefficient for this sample rate!")
		self.warned_about_ineff = true
	end
	assert(action:nElement()==self.lastAction:nElement(),"Inconsistent action sizes")
	for i=1,action:nElement() do
		for j=1,math.max(1,math.floor(resolution/self.sample_rate)) do
			self.lastAction[i] 	= self.lastAction[i] + self.theta*(self.mean[i]-self.lastAction[i])/resolution + self.sigma*torch.randn(1)[1]
		end
	end
	if self.bounds then
		self.lastAction:clamp(self.bounds[1],self.bounds[2])
	else
		self.lastAction:clamp(-1,1)
	end
	action:copy(self.lastAction)	
end

function pol_OU:reset()
	self.lastAction:zero()
end
-- Uniformly random discrete exploration --------------------------------------
local pol_dr = torch.class('drl_discrete_policy_uniform_random')

function pol_dr:__init(settings)
	assert(settings.action_dimension,"action_dimension not set")
	if not(type(settings.action_dimension)=='number') then
		assert(#settings.action_dimension==1,'Only one dimensional discrete actions supported for now')
		self.nractions = settings.action_dimension[1]
	else
		self.nractions = settings.action_dimension
	end
end

function pol_dr:get_for_index(action,time_index)
	if (type(action)=='number') then
		action = math.random(self.nractions)
	elseif (action:nElement() == 1 ) then
		action:copy(torch.Tensor({math.random(self.nractions)}))	
	elseif (action:nElement() == self.nractions ) then
		temp = action:clone():zero()
		temp[math.random(self.nractions)] = 1
		action:copy(temp)
	else
		print('action:')
		print(action)
		assert(false,'Given action is not of the right dimensions ')
	end
end

function pol_dr:reset()
	--
end


--- continuous random
local pol_cr = torch.class('drl_continuous_policy_uniform_random')

function pol_cr:__init(settings)
	assert(settings.action_dimension,"action_dimension not set")
	if not(type(settings.action_dimension)=='number') then
		assert(#settings.action_dimension==1,'number of action dimensions should be R1')
		self.nractions = settings.action_dimension[1]
	else
		self.nractions = settings.action_dimension
	end
	assert(settings.upper_limits and #settings.upper_limits==self.nractions, "upper_limits not set or size not equal to action_dimension")
	assert(settings.lower_limits and #settings.lower_limits==self.nractions, "lower_limits not set or size not equal to action_dimension")
	self.low 	=	settings.lower_limits
	self.high = settings.upper_limits
end

function pol_cr:get_for_index(action,time_index)
	if (action:nElement() == self.nractions ) then
		for i=1,self.nractions do
			action[i] = (math.random()*(self.high[i]-self.low[i]))+self.low[i]
		end
	else
		print('action:')
		print(action)
		assert(false,'Given action is not of the right dimensions ')
	end
end

function pol_cr:reset()
	--
end

local pol_crn = torch.class('drl_continuous_policy_normal_random')

function pol_crn:__init(settings)
	assert(settings.action_dimension,"action_dimension not set")
	if not(type(settings.action_dimension)=='number') then
		assert(#settings.action_dimension==1,'number of action dimensions should be R1')
		self.nractions = settings.action_dimension[1]
	else
		self.nractions = settings.action_dimension
	end
	assert(settings.upper_limits and #settings.upper_limits==self.nractions, "upper_limits not set or size not equal to action_dimension")
	assert(settings.lower_limits and #settings.lower_limits==self.nractions, "lower_limits not set or size not equal to action_dimension")
	self.low 	=	settings.lower_limits
	self.high = settings.upper_limits
end

function pol_crn:get_for_index(action,time_index)
	if (action:nElement() == self.nractions ) then
		for i=1,self.nractions do
			action:copy(torch.randn(action:size()))
		end
	else
		print('action:')
		print(action)
		assert(false,'Given action is not of the right dimensions ')
	end
end

function pol_crn:reset()
	--
end



local expa 	= torch.class('drl_exploration_amount_function')

function expa:__init(settings)
	self.settings = settings	
end 

function expa:__call(time_index,sequence_index)
	assert(self.settings.functiontype,"No functiontype specifiied")
	if (self.settings.functiontype == "constant") then
		assert(self.settings.initial_exploration,"initial_exploration not set")
		return math.max(0,math.min(1,self.settings.initial_exploration))
	elseif (self.settings.functiontype == "exponential_per_sequence") then
		assert(sequence_index,"sequence_index is nil!") 
		assert(self.settings.initial_exploration,"initial_exploration not set")
		assert(self.settings.multiplier,"multiplier not set")
		assert(self.settings.minimum_exploration,"minimum_exploration not set")
		return math.max(self.settings.minimum_exploration,math.min(1,self.settings.initial_exploration*math.pow(self.settings.multiplier,(sequence_index-1))))
	elseif (self.settings.functiontype == "linear_per_sequence") then
		assert(sequence_index,"sequence_index is nil!") 
		assert(self.settings.initial_exploration,"initial_exploration not set")
		assert(self.settings.multiplier,"multiplier not set")
		assert(self.settings.minimum_exploration,"minimum_exploration not set")		
		return math.max(self.settings.minimum_exploration,math.min(1,self.settings.initial_exploration-(math.abs(self.settings.multiplier)*(sequence_index-1))))
	else
		assert(false,"unknown functiontype")
	end 
end









