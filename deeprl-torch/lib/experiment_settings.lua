--[[
		Tim de Bruin 2015
		deepRL-torch
		experiment settings
		Contains all settings needed in an experiment for both the executor and the trainer
--]]


local settingsclass = torch.class('experiment_settings')

function settingsclass:__init()
	self.EN = {} -- settings related to the environment
	self.EX = {} -- settings related to the experience replay database and memory
	self.NN = {} -- settings related to the neural networks
	self.RL = {} -- settings related to the reinforcement learning algorithm		
end
	
function settingsclass:loadFromFile()
	
end

-- Basic DDPG setup
function settingsclass:setDefault()
	self.EN.communicator_settings = {
		sampling_time 	= 0.1, -- seconds
		floatformat 	= "%+1.7e",
		integerformat 	= "%+i",	
	}
	
	
	
	
	
	
	self.EX.short_term_memory_size = 100
	self.EX.experience_replay_size = 20000
	self.EX.RL_state_parts = {state = true, action = true, reward = true, next_state = true, next_action = false}
	self.EX.full_state_dimension 	= {} -- for multi modal states more than one entry can be used
	table.insert(self.EX.full_state_dimension,torch.LongStorage({8}))
	self.EX.action_dimension 		= {} -- multiple outputs can be usefull when one network is used for seperate tasks
	table.insert(self.EX.action_dimension,torch.LongStorage({2}))

end
