require "drl_neural_networks"

local dqn,parent = torch.class('drl_dqn','drl_neural_networks')

function dqn:__init(controller,trainer,settings)
	self.lastTDE = math.huge
	parent.__init(self,controller,trainer,settings)
  --TODO: move to parent?
	if (settings.state_bounds) then
    self.input_scale = torch.Tensor(settings.state_bounds[2]:size()):copy(settings.state_bounds[2]) -- max
    self.input_scale:csub(settings.state_bounds[1]) -- (max-min)
    self.input_scale:mul(0.5)
    self.input_scale_add = torch.Tensor(settings.state_bounds[2]:size()):copy(settings.state_bounds[2]) -- max
    self.input_scale_add:add(settings.state_bounds[1]) -- (max+min)
    self.input_scale_add:mul(0.5)
    self.state_bounds = settings.state_bounds
  end
  -- TODO: move to parent?
  self.updateCount = 0
end

function dqn:reset()
	self.train_networks.Q.network:reset()
	self.controller_network.network:reset()
	self.train_networks.FROZENQ.network:reset()
end

function dqn:create_network(settings)
	assert(settings,"No settings given")
	assert(settings.statesize,"No state size given")
	assert(settings.actionsize,"No action size given")
	assert(settings.hsizes,"size of hidden layer(s) not given")
	assert(torch.type(settings.hsizes)=="table","hsize should be a table array with the number of hidden units per layer")
	local inputsize_state, actionsize

	if (type(settings.statesize) == 'number') then
		inputsize_state = settings.statesize
	elseif #settings.statesize == 1 then
		inputsize_state = settings.statesize[1]
	else
		print("state size:")
		print(settings.statesize)
		assert(false,'not yet implemented')
	end

	if (type(settings.actionsize) == 'number') then
		actionsize = settings.actionsize
	elseif #settings.actionsize == 1 then
		actionsize = settings.actionsize[1]
	else
		assert(false,'not yet implemented')
	end

	local hiddensizes				= settings.hsizes
	local nonlinearity			= settings.nonlinearity or nn.ReLU
	local batchnorm					= nn.Identity
	if settings.batchnorm	then
		batchnorm = nn.BatchNormalization
	end

  local stateInput 				=	nn.Identity()()
	local hiddenlayers 			= {}
	local layersizes				= {}
	hiddenlayers[0] 				= stateInput
	layersizes[0] 					= inputsize_state

	local layerindex = 0
	while layerindex < #hiddensizes do
		layerindex 								= layerindex + 1

		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= hiddensizes[layerindex]
	end
	local Qoutput 							= nn.Linear(layersizes[layerindex],actionsize)(hiddenlayers[layerindex])
	local network 							= nn.gModule({stateInput},{Qoutput})
	return network
end


function dqn:create_train_networks()
	parent.create_train_networks(self)
	self.train_networks.Q 							 			= {}
	self.train_networks.Q.network			 				= self:create_network(self.settings)
	self.train_networks.FROZENQ 							= {}
	self.train_networks.FROZENQ.network				= self:create_network(self.settings)
	self.train_networks.criterion 						= nn.MSECriterion()

	-- first optionally send to GPU, only THEN get the parameters
	if self.settings.GPU then
		self.train_networks.Q.network:cuda()
		self.train_networks.FROZENQ.network:cuda()
		self.train_networks.criterion:cuda()
	end

	local Q_paramx, Q_paramdx 								= self.train_networks.Q.network:getParameters()
	local FQ_paramx, FQ_paramdx 							= self.train_networks.FROZENQ.network:getParameters()
	self.optimStateQ													= {}
	self.train_networks.Q.paramx 							= Q_paramx
	self.train_networks.Q.paramdx 						= Q_paramdx
	self.train_networks.FROZENQ.paramx 				= FQ_paramx
	self.train_networks.FROZENQ.paramdx 			= FQ_paramdx

	self.optimStateQ 													= {}
end

-- Can be either a policy network or a Q network with discrete actions
-- After this call self.controller_network should at least contain:
-- { network, paramx, paramdx }
function dqn:create_controller_network()
	self.controller_network 						= {}
	self.controller_network.network 		= self:create_network(self.settings)
	self.controller_network.paramx 			= self.controller_network.network:getParameters()
end

--- Returns the latest trained parameters for the controller network (from the training network(s))
function dqn:get_controller_parameters()
	return self.train_networks.Q.paramx:clone()
end

function dqn:set_controller_parameters(controller_parameters)
	self.controller_network.paramx:copy(controller_parameters:type(self.controller_network.paramx:type()))
	self.controller_network.network:evaluate()
end

function dqn:get_policy_action(state)
	assert(#state:size()==1,"Improve the following line to cope with multidimensional states")
	local s = state:clone():resize(1,state:nElement())[1]

	if self.input_scale then
  	s:add(-1,self.input_scale_add)
  	s:cdiv(self.input_scale)
  end

  local a = self.controller_network.network:forward(s)
  local max, indices = torch.max(a,1)
  -- return indices[1]
  -- return as a one hot representation:
  aoh = a:clone():zero()
  aoh[indices[1]] = 1
  return aoh
end

local same_dimensions = function(tensor1, tensor2)
  if (not tensor1) or (not tensor2) then
    return false
  end
  if not(#tensor1:size() == #tensor2:size()) then
    return false
  end
  for d = 1,#tensor1:size() do
    if not(tensor1:size(d)==tensor2:size(d)) then
      return false
    end
  end
  return true
end

function dqn:train(experiencedb, trainsettings)
  parent:train(experiencedb, trainsettings)
	local GPU 						= self.train_networks.Q.paramx:type()=="torch.CudaTensor"
	local MINIMIZEQ 			= experiencedb.cost
	self.trainbatch				= self.experiencedb:get_mini_batch({
		batch_size 			= trainsettings.batch_size,
		maxbatches			= trainsettings.batchupdates,
		GPU 						= GPU,
		requested_parts = trainsettings.requested_parts,
		seq_properties  = trainsettings.seq_properties,
		prioritized 		= trainsettings.prioritized_experience_replay,
		prioritized_alpha = trainsettings.prioritized_alpha,
		prioritized_beta 	= trainsettings.prioritized_beta,
		countbasedimpsamp = trainsettings.countbasedimpsamp
	})
  self.updateCount      = self.updateCount + 1

  -- Input scale --
  if (self.state_bounds) then
    if (not same_dimensions(self.batchscale_sub_in, self.trainbatch.state[1])) then
      self.batchscale_sub_in =  self.trainbatch.state[1]:clone()
      for i = 1,self.state_bounds:size(2) do

        self.batchscale_sub_in[{{},{},{},i}]:fill((self.state_bounds[2][i] + self.state_bounds[1][i])/2) -- (max+min)/2
      end
      self.batchscale_mul_in = self.trainbatch.state[1]:clone()
      for i = 1,self.state_bounds:size(2) do
        self.batchscale_mul_in[{{},{},{},i}]:fill(2/(self.state_bounds[2][i] - self.state_bounds[1][i])) -- 2/(max-min)
      end
    end
    self.trainbatch.state[1]:add(-1,self.batchscale_sub_in)
    self.trainbatch.state[1]:cmul(self.batchscale_mul_in)
    self.trainbatch.next_state[1]:add(-1,self.batchscale_sub_in)
    self.trainbatch.next_state[1]:cmul(self.batchscale_mul_in)
  end

	if not trainsettings.state_index then
		if #self.trainbatch.state == 1 then
			trainsettings.state_index = 1
		end
	end
	if ((not trainsettings.seq_properties) or trainsettings.seq_properties.length <=1) then
		assert(trainsettings.state_index,"state_index not set and multiple states available")
		assert(trainsettings.batchupdates,"batchupdates not set (number of minibatch updates per train() call)")
		assert(trainsettings.gamma,"gamma RL parameter not set (dqn uses TD learning)")
		trainsettings.minq = MINIMIZEQ

		for i=1,trainsettings.batchupdates do
			batch_index = math.max(1,i%self.trainbatch.state[1]:size(1))
			self:train_noseq(batch_index,trainsettings.state_index,trainsettings.gamma,trainsettings, experiencedb)
		end
	else
		assert(false, "dqn not implemented for sequences")
	end
end



function dqn:train_noseq(batch_index,state_index,gamma,train_settings, experiencedb)
	local sequence_index = 1
	if self.gradientupdatecount then
		self.gradientupdatecount = self.gradientupdatecount + 1
	else
		self.gradientupdatecount = 1
	end

		-- make sure s, s' and a are scaled!
		function evaluateQ(paramx_)
			if paramx_ ~= self.train_networks.Q.paramx then self.train_networks.Q.paramx:copy(paramx_) end
			--FP

			local s 	    	= self.trainbatch.state[state_index][{batch_index,sequence_index}]
			local a		    	= self.trainbatch.action[state_index][{batch_index,sequence_index}]
			local next_s  	= self.trainbatch.next_state[state_index][{batch_index,sequence_index}]
			local r 	    	= self.trainbatch.reward[{batch_index,sequence_index}]
			local term
			if self.trainbatch.terminal then
				term = self.trainbatch.terminal[{batch_index,sequence_index}]:clone()
			else
				term = r:clone():zero()
			end
			-- to use for the Q(s') in a multiplicative way
			term:mul(-1):add(1)
			local maxqnexts = self.train_networks.FROZENQ.network:forward(next_s):max(2)
			local nextq = maxqnexts:clone():cmul(term)
			local target = torch.add(r,gamma,nextq)

			if cutorch then
				cutorch:synchronize()
			end

      local Qpreds 		= self.train_networks.Q.network:forward(s)
			local q = torch.Tensor(Qpreds:size(1)):typeAs(Qpreds)
			local actionbatchindex = torch.Tensor(Qpreds:size(1))
			if torch.isTensor(a[1]) then
				for i=1,q:nElement() do
					local v,w = torch.max(a[i],1)
					actionbatchindex[i] = w[1]
				end
			else
				actionbatchindex = a:clone()
			end
			for i=1,q:nElement() do
					q[i] = Qpreds[i][actionbatchindex[i]]
			end
			if cutorch then
				cutorch:synchronize()
			end


			--BP

			self.lastTDE 	= self.train_networks.criterion:forward(q,target)
			if train_settings.logTDE then
				--calculate temporal difference error per database sample
				local TDEtens = torch.Tensor(target:size()):type(q:type())
				TDEtens:add(q,-1.0,target:clone():resizeAs(q))
				experiencedb:update_extra_info("TDE",TDEtens:clone():double():abs(),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end
      if train_settings.logQPred then
				experiencedb:update_extra_info("QPRED",q:clone():double(),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end
      if train_settings.logQFpred then
				experiencedb:update_extra_info("QFPRED",maxqnexts:clone():double(),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end
			-- logs the last time (itertion) in which the samples were used for a gradient calculation
      if train_settings.logUPDATE then
				experiencedb:update_extra_info("UPDATE",self.trainbatch.db_indices[{batch_index,sequence_index}]:clone():fill(self.updateCount),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end



			-- Gradient calculation
			self.train_networks.Q.paramdx:zero()
			if cutorch then
				cutorch:synchronize()
			end
			local errorgrad 		= self.train_networks.criterion:backward(q,target)
			local Qgradient 		= torch.Tensor(Qpreds:size()):typeAs(Qpreds):zero()
			for i = 1,a:size(1) do
				Qgradient[i][actionbatchindex[i]] = errorgrad[i]
			end

			self.train_networks.Q.network:backward(s,Qgradient)
			if cutorch then
				cutorch:synchronize()
			end

			-- detect nan gradient
			if (self.train_networks.Q.paramdx:sum() ~= self.train_networks.Q.paramdx:sum()) then
				print("dqn: NaN gradient!")
				self.train_networks.Q.paramdx:zero()
			end
			-- l2 reg
			if self.trainsettings.L2norm then
				self.train_networks.Q.paramdx:add( self.train_networks.Q.paramx:clone():mul( self.trainsettings.L2norm ) )
			end
			return 0 , self.train_networks.Q.paramdx
		end

	-- Start of the training function --
	optim_settings = train_settings.optimsettings
	self.train_networks.Q.network:training()

	assert(optim_settings,"optim_settings not set")
	assert(optim_settings.optimfunction,"optimfunction not set")
	assert(optim_settings.config,"config not set")
	assert(optim_settings.freezecount,"freezecount not set (number of gradient updates before updating the weights of the network used for the target Q values.)")
	optim_settings.optimfunction(evaluateQ, self.train_networks.Q.paramx, optim_settings.config, self.optimStateQ)

	if (self.gradientupdatecount % optim_settings.freezecount == 1) then
		self.train_networks.FROZENQ.paramx:copy(self.train_networks.Q.paramx)
	end
	-- ensure the frozen networks are always on evaluate (but only after having gotten initial params)
	self.train_networks.FROZENQ.network:evaluate()
end

function dqn:getQ(state, action)
  state = state:resize(state:nElement()/self.settings.statesize,self.settings.statesize):typeAs(self.train_networks.Q.paramx)
  action = action:resize(action:nElement()/self.settings.actionsize,self.settings.actionsize):typeAs(self.train_networks.Q.paramx)
  self.train_networks.Q.network:evaluate()
  if (self.state_bounds) then
    if (not same_dimensions(self.qscalein_sub, state)) then
      self.qscalein_sub =  state:clone()
      for i = 1,self.state_bounds:size(2) do
        self.qscalein_sub[{{},i}]:fill((self.state_bounds[2][i] + self.state_bounds[1][i])/2) -- (max+min)/2
      end
      self.qscalein_mul = state:clone()
      for i = 1,self.state_bounds:size(2) do
        self.qscalein_mul[{{},i}]:fill(2/(self.state_bounds[2][i] - self.state_bounds[1][i])) -- 2/(max-min)
      end
    end
    state:add(-1,self.qscalein_sub)
    state:cmul(self.qscalein_mul)
  end
	return self.train_networks.Q.network:forward(state)
end


function dqn:valQTDE(state, next_state, action, reward)
  state = state:resize(state:nElement()/self.settings.statesize,self.settings.statesize):typeAs(self.train_networks.Q.paramx)
  action = action:resize(action:nElement()/self.settings.actionsize,self.settings.actionsize):typeAs(self.train_networks.Q.paramx)
  next_state = next_state:resize(next_state:nElement()/self.settings.statesize,self.settings.statesize):typeAs(self.train_networks.Q.paramx)
  action = action:resize(action:nElement()/self.settings.actionsize,self.settings.actionsize):typeAs(self.train_networks.Q.paramx)
  self.train_networks.Q.network:evaluate()
  if (self.state_bounds) then
    if (not same_dimensions(self.qscalein_sub, state)) then
      self.qscalein_sub =  state:clone()
      for i = 1,self.state_bounds:size(2) do
        self.qscalein_sub[{{},i}]:fill((self.state_bounds[2][i] + self.state_bounds[1][i])/2) -- (max+min)/2
      end
      self.qscalein_mul = state:clone()
      for i = 1,self.state_bounds:size(2) do
        self.qscalein_mul[{{},i}]:fill(2/(self.state_bounds[2][i] - self.state_bounds[1][i])) -- 2/(max-min)
      end
    end
    state:add(-1,self.qscalein_sub)
    state:cmul(self.qscalein_mul)
    next_state:add(-1,self.qscalein_sub)
    next_state:cmul(self.qscalein_mul)
  end

	local next_qs = self.train_networks.Q.network:forward(state)
	nqvs, besta = torch.max(next_qs,2)
	local Qs =  self.train_networks.Q.network:forward(state)
			--self.train_networks.Q.network:evaluate()
end

local copytable = function (table)
	result = {}
	for name, object in pairs(table) do
		if type(object == 'number') then
			result[name] = object
		elseif type(object) == 'table' then
			result[name] = copytable(object)
		else
			result[name] = object:clone()
		end
	end
	return result
end

function dqn:createCheckpoint()
	self.checkpoint = {}
	self.checkpoint.fqp = self.train_networks.FROZENQ.paramx:clone()
	self.checkpoint.qp 	= self.train_networks.Q.paramx:clone()
	self.checkpoint.cp  = self.controller_network.paramx:clone()
	self.checkpoint.optimState = copytable(self.optimStateQ)
	self.checkpoint.updateCount = self.updateCount
	self.checkpoint.gradientupdatecount = self.gradientupdatecount
end

function dqn:resetToLastCheckpoint()
	assert(self.checkpoint,"No checkpoint saved")
	self.train_networks.FROZENQ.paramx:copy(self.checkpoint.fqp)
	self.train_networks.Q.paramx:copy(self.checkpoint.qp)
	self.controller_network.paramx:copy(self.checkpoint.cp)
	self.optimStateQ = copytable(self.checkpoint.optimState)
	self.updateCount = self.checkpoint.updateCount
	self.gradientupdatecount = self.checkpoint.gradientupdatecount
end
