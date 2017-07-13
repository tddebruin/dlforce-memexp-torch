--[[

		Tim de Bruin 2016
		deepRL-torch
		experience database - distribution class.
		Uses SxA samples to train a classifier that determines the probability that a sample is from the policy distribution.
		
--]]

require 'nn'
require 'nngraph'

local classifier = torch.class('drl_experience_classifier')

function classifier:__init( experience_memory, settings )
	assert(settings, 'No settings given')

end

local makeClassifierNetworks = function ( settings )
	assert(settings,'No settings given ')
	self.networks = {}
	if (settings.stateaction) then
		self.networks.stateaction = create_distribution_classifier_network(true, true, settings)
	end	
	if (settings.state) then
		self.networks.stateaction = create_distribution_classifier_network(true, true, settings)
	end	
	if (settings.action) then
		self.networks.stateaction = create_distribution_classifier_network(true, true, settings)
	end	
	self.networks.criterion 	= nn.BCECriterion()
end

local create_distribution_classifier_network = function(useState, useAction, settings)
	assert(settings,"No settings given")
	local nonlinearity	= settings.nonlinearity or nn.ReLU
	assert(settings.hsizes,"size of hidden layer(s) not given")
	assert(torch.type(settings.hsizes)=="table","hsize should be a table array with the number of hidden units per layer") 
	local hiddensizes	= settings.hsizes
	local batchnorm						= nn.Identity
	if settings.actor.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
	local input_table = {}
	if useState then 
	  local s                 = nn.Identity()()
	  assert(settings.statesize,"No state size given")
	  table.insert(input_table,s)
	end
	if useAction then
		local a                 = nn.Identity()()
		assert(settings.actionsize,"No action size given")
		table.insert(input_table,a)
	end

	local hiddenlayers 			= {}
	local layersizes				= {}
	
	if ( useState and useAction) then 
		hiddenlayers[0] 				= nn.JoinTable(2)({batchnorm(settings.statesize)(s), batchnorm(settings.actionsize)(a)})
		layersizes[0] 					= settings.statesize	+ settings.actionsize	
	elseif (useState) then
		hiddenlayers[0] 				= batchnorm(settings.statesize)(s)
		layersizes[0] 					= settings.statesize		
	else
		assert(useAction,'Both useState and useAction are false or nil, the sample classification should be based on something...')
		hiddenlayers[0] 				= batchnorm(settings.actionsize)(a)
		layersizes[0] 					= settings.actionsize			
	end

	local layerindex = 0
	while layerindex < #hiddensizes do 
		layerindex 								= layerindex + 1
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= hiddensizes[layerindex]
	end 

  local h2c              = nn.Linear(layersizes[layerindex], 1)(hiddenlayers[layerindex])
  local pred             = nn.Sigmoid()(h2c())

  local module           = nn.gModule(input_table, 
                                      {pred})
  return module
end

-- settings should contain: classifier, dbInideces
local train_classifier = function( network, settings )
	



end

-- bounds: Tensor{2 x n}
local generate_uniform_samples = function(numberOfSamples, bounds)
	synth = torch.Tensor(numberOfSamples, bounds:size(2))
	for i=1,bounds:size(2) do
		synth[{{},i}]:copy( torch.rand(numberOfSamples)*(bounds[2][i]-bounds[1][i]) + bounds[1][i] )
	end
	return synth
end

local sw = torch.class('drl_sliding_window')

function sw:__init( experience_memory, kernelscale, statebounds, actionbounds  )
	if statebounds and #statebounds:size()==2 then self.statedim = statebounds:size(2) else self.statedim = 0 end
	if actionbounds and #actionbounds:size()==2 then self.actiondim = actionbounds:size(2) else self.actiondim = 0 end
	
  self.experiencedb = experience_memory.experience_database
	self.dimensions = self.statedim + self.actiondim
	self.update = 0
	self.state = self.statedim > 0
	self.action = self.actiondim > 0
	self.kernelscale = {state = {}, action = {}}
  if self.state then 
  	self.uniform_states = generate_uniform_samples(self.experiencedb.time_indices:nElement(),statebounds)   	
  	for i = 1,self.statedim do
  		self.kernelscale.state[i] = (statebounds[2][i] - statebounds[1][i])/kernelscale
  	end
  end
	if self.action then 
		self.uniform_actions = generate_uniform_samples(self.experiencedb.time_indices:nElement(),actionbounds) 
  	for i = 1,self.actiondim do
  		self.kernelscale.action[i] = (actionbounds[2][i] - actionbounds[1][i])/kernelscale
  	end		
	end
end 

function sw:update_distributions( nrsamples )	
	--self.sigmadb = { state = {}, action = {}}
	--self.sigmaid = { state = {}, action = {}}
	
	local realnrsamples = math.min(nrsamples,self.experiencedb.last_write_index)
	self.nrsamples =realnrsamples
	if nrsamples < self.experiencedb.last_write_index then
		local ShuffledIndices = torch.randperm( self.experiencedb.last_write_index )
		local randomIndices 		  = ShuffledIndices[{{1 ,  nrsamples }}]:long()
		if self.state then self.dbstates = self.experiencedb.state[1]:index(1,randomIndices) 
		end
		if self.action then self.dbactions = self.experiencedb.action:index(1,randomIndices) 
		end
	else -- use whole db
		if self.state then self.dbstates = self.experiencedb.state[1][{{1,realnrsamples},{}}] 
		end
		if self.action then self.dbactions = self.experiencedb.action[1][{{1,realnrsamples},{}}] 
		end
	end
--[[
	if self.state then 
		self.unistates = self.uniform_states[{{1,realnrsamples},{}}] 
	end
	if self.action then 
		self.uniactions = self.uniform_actions[{{1,realnrsamples},{}}] 
	end
--[[
	for s=1,self.statedim do
		table.insert(self.sigmadb.state,self:calculate_stdv(self.dbstates))
		table.insert(self.sigmaid.state,self:calculate_stdv(self.unistates))
	end
	for s=1,self.actiondim do
		table.insert(self.sigmadb.action,self:calculate_stdv(self.dbactions))
		table.insert(self.sigmaid.action,self:calculate_stdv(self.uniactions))
	end
--]]
end

-- determine which points to overwrite once per episode.
function sw:check_distribution()
	self.sample_dbdistr = torch.Tensor(self.experiencedb.last_write_index)
	for i = 1,self.experiencedb.last_write_index do
		self.sample_dbdistr[i] = self:check_point(self.experiencedb.state[1][i],self.experiencedb.action[1][i])
  end
end

function sw:calculate_stdv( vector )
	--vector:std (rtfm)
end

-- returns the degree of membership of the point to the database distribution and the ideal distribution
function sw:check_point( state, action )
		local total_dist_state_ddb = torch.Tensor(self.nrsamples,1)
		--local total_dist_state_did = torch.Tensor(self.nrsamples,1)
		local total_dist_action_ddb = torch.Tensor(self.nrsamples,1)
		--local total_dist_action_did = torch.Tensor(self.nrsamples,1)
		if self.state then 
			state:resize(1,self.statedim)
      local dist_state_ddb = torch.repeatTensor(state,self.nrsamples,1)
		--	local dist_state_did = torch.repeatTensor(state,self.nrsamples,1)
			dist_state_ddb:add(-1,self.dbstates)
		--	dist_state_did:add(-1,self.unistates)
			dist_state_ddb:pow(2)
		--	dist_state_did:pow(2)
			for i=1,self.statedim do
				dist_state_ddb[{{},i}]:div(self.kernelscale.state[i])
		--		dist_state_did[{{},i}]:div(self.kernelscale.state[i])
			end
			 total_dist_state_ddb = dist_state_ddb:sum(2) 
		--	 total_dist_state_did = dist_state_did:sum(2) 
		else
			 total_dist_state_ddb:zero() 
		--	 total_dist_state_did:zero()
		end
		if self.action then 
      action:resize(1,self.actiondim)
			local dist_action_ddb = torch.repeatTensor(action,self.nrsamples,1)
	--		local dist_action_did = torch.repeatTensor(action,self.nrsamples,1)
			dist_action_ddb:add(-1,self.dbactions)
	--		dist_action_did:add(-1,self.uniactions)
			dist_action_ddb:pow(2)
	--		dist_action_did:pow(2)
			for i=1,self.actiondim do
				dist_action_ddb[{{},i}]:div(self.kernelscale.action[i])
	--			dist_action_did[{{},i}]:div(self.kernelscale.action[i])
			end
			 total_dist_action_ddb = dist_action_ddb:sum(2) 
	--		 total_dist_action_did = dist_action_did:sum(2) 
		else
			 total_dist_action_ddb:zero() 
	--		 total_dist_action_did:zero()
		end
		
		local membership_ddb = total_dist_state_ddb:add(total_dist_action_ddb):mul(-1):exp():sum()/self.nrsamples
	--	local membership_did = total_dist_state_did:add(total_dist_action_did):mul(-1):exp():sum()/self.nrsamples

		return membership_ddb--,membership_did		
end



--points are tables with state and action fields optionally set
-- returns true if the second point is more desirable then the first
function sw:compare_points(sequencenr,point1,point2)
	if self.update < sequencenr then
		self.update = sequencenr
		self.update_distributions(math.huge)
	end
	local ddb1, did1 = self:check_point(point1.state, point1.action)
	local ddb2, did2 = self:check_point(point2.state, point2.action)
	return ((did1/ddb1) < (did2/ddb2))
end

--[[
function sw:find_worst_experience_index()
	worstValue = math.huge
	worstIndex = 0
	for i = 1,self.experiencedb.last_write_index do
		ddb, did = self:check_point(self.experiencedb.state[1][i],self.experiencedb.action[1][i])
		local val = did/ddb 
		if val < worstValue then
			worstValue = val
			worstIndex = i 
		end
  end
	return worstIndex
end
--]]

function sw:find_worst_experience_index()
	worstValue = 0
	worstIndex = 0
	for i = 1,self.experiencedb.last_write_index do
		ddb = self:check_point(self.experiencedb.state[1][i],self.experiencedb.action[1][i])
		if ddb > worstValue then
			worstValue = ddb
			worstIndex = i 
		end
  end
	return worstIndex
end

function sw:return_worst_experience_index()
	-- worst index is the one most like the DB distribution
	dens, index = self.sample_dbdistr:max(1)
	self.sample_dbdistr[index[1]] = 0
	return index[1]
end