--[[

		Tim de Bruin 2016
		deepRL-torch
		Fantasy experience model check class 
		
		
--]]

require 'nn'
require 'nngraph'

local dmc = torch.class('drl_daydream_modelcheck')
--[[
settings{
	checks = {
		!optional: dynamics [boolean]
		!optional: reward [boolean]
		!optional: terminal state [boolean]
	}
	daydream_modules - {
		(table of name, drl_dayrdeam), one or more daydream modules.
	}
	primary_database = drl_experience_database: database used to train the daydream modules, with the train, validate and test fractions set and all greater than 0.
	!optional: secondary_database = drl_experience_database: 
	state_dimension
	action_dimension
}

]]

 -- TODO: add max?
function dmc:__init( settings )
	local INIT_TENSOR_SIZE = 1000
	assert(settings, 'No settings given')
	self.lastIndex 				= 0
	self.checks 					= settings.checks
	self.daydream_modules = settings.daydream_modules
	self.PDB 							= settings.primary_database
	self.SDB 							= settings.secondary_database
	self.results 					= { samples = torch.Tensor(INIT_TENSOR_SIZE):zero(), seq_idx = torch.Tensor(INIT_TENSOR_SIZE):zero() }
	self.MSECriterion 		= nn.MSECriterion()
	self.BCECriterion 		= nn.BCECriterion()	
	self.VAF_k						= settings.VAF_k
	self.performance_measure = settings.perf_function

	for name, mod in pairs(self.daydream_modules) do
		self.results[name] = {}
		if self.checks.dynamics then
			self.results[name].dynamics = {}
			self.results[name].dynamics.training = torch.Tensor(INIT_TENSOR_SIZE):zero()
			self.results[name].dynamics.validation = torch.Tensor(INIT_TENSOR_SIZE):zero()
			self.results[name].dynamics.VAF 	= torch.Tensor(INIT_TENSOR_SIZE):zero()
			self.results[name].dynamics.DVAF 	= torch.Tensor(INIT_TENSOR_SIZE):zero()
			if self.VAF_k then 
				self.results[name].dynamics.VAFK 		= torch.Tensor(self.VAF_k,INIT_TENSOR_SIZE):zero()
				self.results[name].dynamics.DVAFK 	= torch.Tensor(self.VAF_k,INIT_TENSOR_SIZE):zero()
			end
			self.results[name].dynamics.ACOR 	= torch.Tensor(INIT_TENSOR_SIZE,settings.state_dimension[1],settings.action_dimension[1]):zero()
			self.results[name].dynamics.ACORN 	= torch.Tensor(INIT_TENSOR_SIZE,settings.state_dimension[1],settings.action_dimension[1]):zero()
			self.results[name].dynamics.selftest = torch.Tensor(INIT_TENSOR_SIZE):zero()
			if self.SDB then
				self.results[name].dynamics.externaltest = torch.Tensor(INIT_TENSOR_SIZE):zero()
			end
		end
		if self.checks.performancedelta then
			self.results[name].performancedelta = {}
			self.results[name].performancedeltaSTDV = {}
			for idx, rollout in pairs(self.checks.performancedelta) do
				self.results[name].performancedelta['rollout_' .. rollout] = torch.Tensor(INIT_TENSOR_SIZE):zero()
				self.results[name].performancedeltaSTDV['rollout_' .. rollout] = torch.Tensor(INIT_TENSOR_SIZE):zero()
			end		
		end
		if self.checks.reward then
			self.results[name].reward = {}
			self.results[name].reward.training = torch.Tensor(INIT_TENSOR_SIZE):zero()
			self.results[name].reward.validation = torch.Tensor(INIT_TENSOR_SIZE):zero()
			self.results[name].reward.selftest = torch.Tensor(INIT_TENSOR_SIZE):zero()
			if self.SDB then
				self.results[name].reward.externaltest = torch.Tensor(INIT_TENSOR_SIZE):zero()
			end
		end
		if self.checks.terminal then
			self.results[name].terminal = {}
			self.results[name].terminal.training = torch.Tensor(INIT_TENSOR_SIZE):zero()
			self.results[name].terminal.validation = torch.Tensor(INIT_TENSOR_SIZE):zero()
			self.results[name].terminal.selftest = torch.Tensor(INIT_TENSOR_SIZE):zero()
			if self.SDB then
				self.results[name].terminal.externaltest = torch.Tensor(INIT_TENSOR_SIZE):zero()
			end
		end
	end
	if self.checks.performancedelta then
		self.results.realxp = {performancedelta = {}, performancedeltaSTDV = {}}
		self.results.realxp.performancedelta.no_update = torch.Tensor(INIT_TENSOR_SIZE):zero()
		self.results.realxp.performancedeltaSTDV.no_update = torch.Tensor(INIT_TENSOR_SIZE):zero()
		self.results.realxp.performancedelta.db_update = torch.Tensor(INIT_TENSOR_SIZE):zero()
		self.results.realxp.performancedeltaSTDV.db_update = torch.Tensor(INIT_TENSOR_SIZE):zero()
	end
end

local check_tensor_index = function (tensor,index,value)
	if tensor:size(1) <= index then
		local sze = tensor:size():totable()
		sze[1] = sze[1] * 2
		local newtensor = torch.Tensor(unpack(sze)):zero()
		newtensor[{{1,tensor:nElement()}}]:copy(tensor)
		tensor = newtensor
	end
end

local write_to_tensor = function (tensor,index,value)
	check_tensor_index(tensor,index,value)
	if type(value)=='number' then
		tensor[index] = value
	else
		tensor[index]:copy(value)
	end
end

local add_to_tensor = function (tensor, index, value)
	check_tensor_index(tensor,index,value)
	if type(value)=='number' then
		tensor[index] = tensor[index] + value
	else
		tensor[index]:add(value)
	end
end

function dmc:run_tests(data_index, sequence_index)
	self.lastIndex = self.lastIndex + 1
	if (data_index) then
		write_to_tensor(self.results.samples,self.lastIndex,data_index)
	else
		write_to_tensor(self.results.samples,self.lastIndex,-1)
	end
	if (sequence_index) then
		write_to_tensor(self.results.seq_idx,self.lastIndex, sequence_index )
	else
		write_to_tensor(self.results.seq_idx,self.lastIndex, -1)
	end

	local lastTest = 3
	if self.SDB then
		lastTest = 4
	end

	for test = 1, lastTest do
		local database, dataloadname, dataname
		if test == 1 then
			database 			= self.PDB
			dataloadname 	= "train"
			dataname 			= "training"	
		elseif test == 2 then
			database 			= self.PDB
			dataloadname 	= "validate"
			dataname 			= "validation" 
		elseif test == 3 then
			database 			= self.PDB
			dataloadname 	= "test"
			dataname 			= "selftest"
		elseif test == 4 then
			database 			= self.SDB
			dataloadname 	= nil
			dataname 			= "externaltest"
		end


		local dataset			= database:get_mini_batch(25, false, nil, nil, false, dataloadname)
		for minibatch_counter = 1,dataset.db_indices:size(1) do
			local statepart = 1
			local timeseqidx = 1
			local state = dataset.state[statepart][minibatch_counter][timeseqidx]
			local action = dataset.action[statepart][minibatch_counter][timeseqidx]
			local next_state = dataset.next_state[statepart][minibatch_counter][timeseqidx]
			local reward = dataset.reward[minibatch_counter][timeseqidx]
			local terminal
			if dataset.terminal then
				terminal = dataset.terminal[minibatch_counter][timeseqidx]:clone()
			else
				terminal = reward:clone():zero()
			end			

			if self.checks.dynamics then
				for modelname, model in pairs(self.daydream_modules) do
					local predsprime = model:predict_next_state(state,action)
					local mse = self.MSECriterion:forward(predsprime,next_state)
					add_to_tensor(self.results[modelname].dynamics[dataname], self.lastIndex, mse)			
					
					if dataname == 'validation' then
						local prederr 	= predsprime:clone():csub(next_state)
						local prederrsq = torch.sum(torch.pow(prederr,2),1)
						local nextssq 	= torch.sum(torch.pow(next_state,2),1)
						add_to_tensor(self.results[modelname].dynamics['VAF'], self.lastIndex, torch.sum(torch.clamp(torch.cdiv(prederrsq,nextssq),0,1)))
						local stepchangesq = torch.sum(torch.pow(state:clone():csub(next_state),2),1)
						add_to_tensor(self.results[modelname].dynamics['DVAF'], self.lastIndex, torch.sum(torch.clamp(torch.cdiv(prederrsq,stepchangesq),0,1)))
						
						prederr:resize(state:size(1),state:size(2),1)
						prederra = torch.abs(prederr)
						local actionrs = action:clone():resize(action:size(1),1,action:size(2))
						actiona	 = torch.abs(actionrs)
						add_to_tensor(self.results[modelname].dynamics['ACOR'], self.lastIndex, torch.sum(torch.bmm(prederr,actionrs),1)[1])
						add_to_tensor(self.results[modelname].dynamics['ACORN'], self.lastIndex, torch.sum(torch.bmm(prederra,actiona),1)[1])
					end	

					if minibatch_counter == dataset.db_indices:size(1) then
						self.results[modelname].dynamics[dataname][self.lastIndex] = self.results[modelname].dynamics[dataname][self.lastIndex] / minibatch_counter
						if dataname == 'validation' then
							self.results[modelname].dynamics['VAF'][self.lastIndex] = 1 - (self.results[modelname].dynamics['VAF'][self.lastIndex] / (minibatch_counter * state:size(1)))
							self.results[modelname].dynamics['DVAF'][self.lastIndex] = 1 - (self.results[modelname].dynamics['DVAF'][self.lastIndex] / (minibatch_counter * state:size(1)))
							self.results[modelname].dynamics['ACORN'][self.lastIndex]:copy(torch.abs(self.results[modelname].dynamics['ACOR'][self.lastIndex]:cdiv(self.results[modelname].dynamics['ACORN'][self.lastIndex])))
						end
					end
				end	
			end

			if self.checks.reward then
				for modelname, model in pairs(self.daydream_modules) do
					local mse = self.MSECriterion:forward(model:predict_reward(state,action,next_state),reward)
					add_to_tensor(self.results[modelname].reward[dataname], self.lastIndex, mse)
					if minibatch_counter == dataset.db_indices:size(1) then
						self.results[modelname].reward[dataname][self.lastIndex] = self.results[modelname].reward[dataname][self.lastIndex] / minibatch_counter
					end
				end	
			end

			if self.checks.terminal then
				for modelname, model in pairs(self.daydream_modules) do
					local bce = self.BCECriterion:forward(model:predict_terminal(next_state),terminal)
					add_to_tensor(self.results[modelname].terminal[dataname], self.lastIndex, bce)
					if minibatch_counter == dataset.db_indices:size(1) then
						self.results[modelname].terminal[dataname][self.lastIndex] = self.results[modelname].terminal[dataname][self.lastIndex] / minibatch_counter
					end
				end	
			end
		end
	end
	if self.VAF_k then -- special dynamics modeltest with data in multiple sequential timesteps
		assert(self.checks.dynamics , "VAF_k is for the dynamics model but that is not set to be tested.")	
		local BATCH_SIZE = 10
		local dataset = self.PDB:get_mini_batch(BATCH_SIZE,false,nil,{length = self.VAF_k},false)
		local statepart = 1
		for minibatch_counter = 1,dataset.db_indices:size(1) do
			local state = dataset.state[statepart][minibatch_counter]
			local action = dataset.action[statepart][minibatch_counter]
			local next_state = dataset.next_state[statepart][minibatch_counter]
			local prev_pred_state, prederrsqc, stepchangesqc, nextssqc
			for modelname, model in pairs(self.daydream_modules) do
				for timeseqidx = 1, self.VAF_k do
					if timeseqidx == 1 then 
						prev_pred_state = state[1]:clone()
					end
					local predsprime = model:predict_next_state(prev_pred_state,action[timeseqidx])
					prev_pred_state:copy(predsprime)
					local prederr 	= predsprime:clone():csub(next_state[timeseqidx])
					local prederrsq = torch.sum(torch.pow(prederr,2),1)
					local nextssq 	= torch.sum(torch.pow(next_state[timeseqidx],2),1)
					local stepchangesq = torch.sum(torch.pow(state[timeseqidx]:clone():csub(next_state[timeseqidx]),2),1)

					if timeseqidx == 1 then 
						prederrsqc 		= prederrsq:clone()
						nextssqc			= nextssq:clone()
						stepchangesqc = stepchangesq:clone()
					else
						prederrsqc:add(prederrsq)
						nextssqc:add(nextssq)
						stepchangesqc:add(stepchangesq)
					end

					add_to_tensor(self.results[modelname].dynamics['VAFK'][timeseqidx], self.lastIndex, torch.sum(torch.clamp(torch.cdiv(prederrsqc,nextssqc),0,1)))
		
					add_to_tensor(self.results[modelname].dynamics['DVAFK'][timeseqidx], self.lastIndex, torch.sum(torch.clamp(torch.cdiv(prederrsqc,stepchangesqc),0,1)))
				end -- for the time indices VAK_k
			end -- for each NN
		end -- for each batch in the DB										
		for modelname, model in pairs(self.daydream_modules) do
			for k = 1,self.VAF_k do
				self.results[modelname].dynamics['VAFK'][k][self.lastIndex] = 1 - (self.results[modelname].dynamics['VAFK'][k][self.lastIndex] / (dataset.db_indices:size(1) * BATCH_SIZE))
				self.results[modelname].dynamics['DVAFK'][k][self.lastIndex] = 1 - (self.results[modelname].dynamics['DVAFK'][k][self.lastIndex] / (dataset.db_indices:size(1) *  BATCH_SIZE))
			end
		end
	end
end

function dmc:run_performance_delta_test( rlnetwork, trainsettings, trainsteps )
	
	

	local average_performance = function(db, tsettings)
		local CHECKS = 5
		local perfTens = torch.Tensor(CHECKS):zero()
		for check = 1,CHECKS do
			if db and tsettings then
				rlnetwork:resetToLastCheckpoint()
				for i = 1, trainsteps do
					rlnetwork:train(db, tsettings)
				end
				rlnetwork:set_controller_parameters(rlnetwork:get_controller_parameters())
			end
			perfTens[check] = self.performance_measure( rlnetwork )
		end
		return {mean = perfTens:mean(), stdev = perfTens:std()}
	end

	-- save network state
	rlnetwork:createCheckpoint()
	-- check current performance
	local perf = average_performance(nil,nil)
	write_to_tensor(self.results.realxp.performancedelta.no_update,self.lastIndex,perf.mean)
	write_to_tensor(self.results.realxp.performancedeltaSTDV.no_update,self.lastIndex,perf.stdev)
	-- run updates on real DB
	
	local perf = average_performance(self.SDB,trainsettings)
	write_to_tensor(self.results.realxp.performancedelta.db_update,self.lastIndex,perf.mean)
	write_to_tensor(self.results.realxp.performancedeltaSTDV.db_update,self.lastIndex,perf.stdev)
	
	-- for each daydream module
	for name, mod in pairs(self.daydream_modules) do
		-- for each number of rollout steps experiment
		for idx, rollout in pairs(self.checks.performancedelta) do
			-- train net
			mod:setDreamParameters('rollout', rollout)
			local perf = average_performance(mod,trainsettings)
			write_to_tensor(self.results[name].performancedelta['rollout_' .. rollout],self.lastIndex,perf.mean)
			write_to_tensor(self.results[name].performancedeltaSTDV['rollout_' .. rollout],self.lastIndex,perf.stdev)
		end		
	end
	rlnetwork:resetToLastCheckpoint()
	rlnetwork:set_controller_parameters(rlnetwork:get_controller_parameters())
end

function dmc:resultsave()
	torch.save('temp_modeltest_save.dat',self)
end

function dmc:resultsave_mat(filename)
	--dmc:resultsave()
	
	matresults = {}
	for modelname, model in pairs(self.results) do
		if modelname == 'samples' or modelname == 'seq_idx' then
			matresults[modelname] = model[{{1,math.max(1,math.min(self.lastIndex, model:nElement()))}}] 	
		else
			for modeltypename, modeltype in pairs(model) do
				for resultname, result in pairs(modeltype) do
					local name = '' .. modelname .. modeltypename .. resultname
					if resultname == 'VAFK' or  resultname == 'DVAFK' then
						matresults[name] = result[{{},{1,math.max(1,math.min(self.lastIndex, result:nElement()))}}] 
					else
						matresults[name] = result[{{1,math.max(1,math.min(self.lastIndex, result:nElement()))}}] 
					end
				end
			end
		end
	end 
	mattorch.save(filename, matresults)
end