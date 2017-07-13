--[[
        Tim de Bruin 2015

--]]

require "nn"
if USEGPU then
	require "cunn"
end
require "nngraph"
require "optim"
require "drl_experience_database"

local neural = torch.class('drl_neural_networks')

function neural:__init(controller,trainer,settings)
	self.settings = settings
	if trainer then
		self:create_train_networks()
	end
	if controller then
		self:create_controller_network()
	end		
end

function neural:create_train_networks()
	self.train_networks = {controllerparamx = torch.Tensor()}

end

-- Can be either a policy network or a Q network with discrete actions
-- After this call self.controller_network should at least contain:
-- { network, paramx, paramdx } 
function neural:create_controller_network()
	self.controller_network = {} -- implement in child class, should be a nnmodule (cpu) 
	
end

--- Returns the latest trained parameters for the controller network (from the training network(s))
function neural:get_controller_parameters()
	return self.train_networks.controllerparamx:clone()
end

function neural:set_controller_parameters(controller_parameters)
	self.controller_network.paramx:copy(controller_parameters:type(self.controller_network.paramx:type()))
end

function neural:get_policy_action(state)
	return self.controller_network.network:forward(state)
end

function neural:train(experiencedb, trainsettings)
	assert(experiencedb.experience_database, "no experience database to train from")
	assert(trainsettings.batch_size,"At least the batch size should be given in the trainsettings table")
	self.experiencedb 		= experiencedb
	self.trainsettings	 	= trainsettings
end

	
	

