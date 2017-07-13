require "drl_neural_networks"

local ddpg,parent = torch.class('drl_ddpg','drl_neural_networks')

function ddpg:__init(controller,trainer,settings)
	print("WARNING: DDPG TODO: Handle final states properly.")
	self.lastTDE = math.huge
	parent.__init(self,controller,trainer,settings)
  --TODO: move to parent?
  if (settings.action_bounds) then
    self.output_scale = torch.Tensor(settings.action_bounds[2]:size()):copy(settings.action_bounds[2]) -- max
    self.output_scale:csub(settings.action_bounds[1]) -- (max-min)
    self.output_scale:mul(0.5)
    self.output_scale_add = torch.Tensor(settings.action_bounds[2]:size()):copy(settings.action_bounds[2]) -- max
    self.output_scale_add:add(settings.action_bounds[1]) -- (max+min)
    self.output_scale_add:mul(0.5)
    self.action_bounds = settings.action_bounds
  end
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

function ddpg:reset()
	self.train_networks.critic.network:reset()
	self.train_networks.actor.network:reset()
	self.controller_network.network:reset()
	self.train_networks.FROZENcritic.network:reset()
	self.train_networks.FROZENactor.network:reset()
end

function ddpg:create_actor_network(settings)
	assert(settings,"No settings given")
	assert(settings.actor,"Specifications of the actor network not given")
	assert(settings.statesize,"No state size given")
	assert(settings.actionsize,"No action size given")
	assert(settings.actor.hsizes,"size of hidden layer(s) not given")
	assert(torch.type(settings.actor.hsizes)=="table","hsize should be a table array with the number of hidden units per layer") 
	
	local inputsize_state 		= settings.statesize
	local outputsize_action 	= settings.actionsize
	local hiddensizes					= settings.actor.hsizes
	local nonlinearity				= settings.actor.nonlinearity or nn.ReLU
	local batchnorm						= nn.Identity
	if settings.actor.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
	
  local stateInput 				=	nn.Identity()()
	local actionInput 			= nn.Identity()()	
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
	local Aoutput 							= nn.Tanh()(nn.Linear(layersizes[layerindex],outputsize_action)(hiddenlayers[layerindex]))
	-- TODO: IF YOU REMOVE OR CHANGE THE TANH, CHANGE THE SCALING AT TWO POINTS IN THIS FILE! SEARCH FOR TANHSCALE
  
  local network 							= nn.gModule({stateInput},{Aoutput})
  -- TODO: Lillicrap:
  -- network.backwardnodes[2].data.module.bias:uniform(-3e-3,3e-3)
	-- network.backwardnodes[2].data.module.weight:uniform(-3e-3,3e-3)		
	return network	
end

function ddpg:create_critic_network(settings)
	assert(settings,"No settings given")
	assert(settings.statesize,"No state size given")
	assert(settings.actionsize,"No action size given")
	assert(settings.critic,"Critic specifications not given")
	assert(settings.critic.hsizes,"size of hidden layer(s) not given")
	assert(torch.type(settings.critic.hsizes)=="table","hsize should be a table array with the number of hidden units per layer") 
	assert(settings.critic.actionlayer and settings.critic.actionlayer >= 0 and settings.critic.actionlayer <= #settings.critic.hsizes,"No or incorrect actionlayer given (set the layer at which the actions are concatenated into the critic network activations. with 0 = together with the state inputs and #hsizes = just before the linear output layer")
	
	local action_layer 			= settings.critic.actionlayer
	local inputsize_state 	= settings.statesize
	local inputsize_action 	= settings.actionsize
	local hiddensizes				= settings.critic.hsizes
	local nonlinearity			= settings.critic.nonlinearity or nn.ReLU
	local batchnorm					= nn.Identity
	if settings.critic.batchnorm	then
		batchnorm = nn.BatchNormalization
	end
	
  local stateInput 				=	nn.Identity()()
	local actionInput 			= nn.Identity()()	
	local hiddenlayers 			= {}
	local layersizes			= {}
	hiddenlayers[0] 			= stateInput
	layersizes[0] 				= inputsize_state		
	
	local layerindex = 0
	while layerindex < action_layer do 
		layerindex 								= layerindex + 1
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],hiddensizes[layerindex])(batchnorm(layersizes[layerindex-1])(hiddenlayers[layerindex-1])))
		layersizes[layerindex] 		= hiddensizes[layerindex]
	end 
	hiddenlayers[layerindex] 		= nn.JoinTable(2)({batchnorm(layersizes[layerindex])(hiddenlayers[layerindex]), actionInput})
	layersizes[layerindex] 			= layersizes[layerindex] + inputsize_action
	while layerindex < #hiddensizes do -- no more batchnormalization after adding the action inputs (Lillicrap)
		layerindex 								= layerindex + 1
		hiddenlayers[layerindex] 	= nonlinearity()(nn.Linear(layersizes[layerindex-1],hiddensizes[layerindex])(hiddenlayers[layerindex-1]))
		layersizes[layerindex] 		= hiddensizes[layerindex]
	end 	
	local Qoutput 							= nn.Linear(layersizes[layerindex],1)(hiddenlayers[layerindex])
	
	local network 							= nn.gModule({stateInput, actionInput},{Qoutput})
	
  -- TODO: Lillicrap:
  -- network.backwardnodes[2].data.module.bias:uniform(-3e-3,3e-3)
	-- network.backwardnodes[2].data.module.weight:uniform(-3e-3,3e-3)		
	return network
end

function ddpg:create_train_networks()
	parent.create_train_networks(self)
	self.train_networks.critic 							 	= {}
	self.train_networks.critic.network			 	= self:create_critic_network(self.settings)
	self.train_networks.actor 				 				= {}
	self.train_networks.actor.network  				= self:create_actor_network(self.settings)
	self.train_networks.FROZENcritic 	 				= {}
	self.train_networks.FROZENcritic.network 	= self:create_critic_network(self.settings)
	self.train_networks.FROZENactor 				 	= {}
	self.train_networks.FROZENactor.network  	= self:create_actor_network(self.settings)
	self.train_networks.criterion 						= nn.MSECriterion() 
	
	-- first optionally send to GPU, only THEN get the parameters
	if self.settings.GPU then
		self.train_networks.critic.network:cuda()
		self.train_networks.actor.network:cuda()		
		self.train_networks.FROZENcritic.network:cuda()
		self.train_networks.FROZENactor.network:cuda()
		self.train_networks.criterion:cuda() 
	end	
	
	local critic_paramx, critic_paramdx 			= self.train_networks.critic.network:getParameters()
	local Fcritic_paramx, Fcritic_paramdx 		= self.train_networks.FROZENcritic.network:getParameters()
	self.train_networks.critic.paramx 				= critic_paramx
	self.train_networks.critic.paramdx 				= critic_paramdx
	self.train_networks.FROZENcritic.paramx 	= Fcritic_paramx
	self.train_networks.FROZENcritic.paramdx 	= Fcritic_paramdx
	
	local actor_paramx, actor_paramdx 				= self.train_networks.actor.network:getParameters()
	local Factor_paramx, Factor_paramdx 			= self.train_networks.FROZENactor.network:getParameters()
	self.train_networks.actor.paramx 					= actor_paramx
	self.train_networks.actor.paramdx 				= actor_paramdx
	self.train_networks.FROZENactor.paramx 		= Factor_paramx
	self.train_networks.FROZENactor.paramdx 	= Factor_paramdx
	
	--self.train_networks.controllerparamx 			= actor_paramx	-- parameters used to update the executive controller
	
	self.optimStateActor											= {}
	self.optimStateCritic											= {}
end

-- Can be either a policy network or a Q network with discrete actions
-- After this call self.controller_network should at least contain:
-- { network, paramx, paramdx } 
function ddpg:create_controller_network()
	self.controller_network 						= {}
	self.controller_network.network 		= self:create_actor_network(self.settings)
	self.controller_network.paramx 			= self.controller_network.network:getParameters()
end

--- Returns the latest trained parameters for the controller network (from the training network(s))
function ddpg:get_controller_parameters()
	return self.train_networks.actor.paramx:clone()
	--return self.train_networks.controllerparamx:clone()
end

function ddpg:set_controller_parameters(controller_parameters)
	self.controller_network.paramx:copy(controller_parameters:type(self.controller_network.paramx:type()))
	
	self.controller_network.network:evaluate()
end

function ddpg:get_policy_action(state)
	assert(#state:size()==1,"Improve the following line to cope with multidimensional states")
	local s = state:clone():resize(1,state:nElement())[1]

  s:add(-1,self.input_scale_add)
  s:cdiv(self.input_scale)

  local a = self.controller_network.network:forward(s)
  --TANHSCALE
  if self.action_bounds then 
    --TODO: OPTIMIZE, WRITTEN LATE AT NIGHT
    
    a:cmul(self.output_scale)
    a:add(self.output_scale_add)
  else
    assert(false)
  end
  return a
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

function ddpg:train(experiencedb, trainsettings)

  
  parent:train(experiencedb, trainsettings)
	local GPU 						= self.train_networks.actor.paramx:type()=="torch.CudaTensor"
	--local GPU 						= self.train_networks.controllerparamx:type()=="torch.CudaTensor"
	local MINIMIZEQ 			= experiencedb.cost
	self.trainbatch				= self.experiencedb:get_mini_batch(trainsettings.batch_size, GPU, trainsettings.requested_parts, trainsettings.seq_properties)
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

	  
  --TANHSCALE-------------------------------------------
  if (self.action_bounds) then 
    if (not same_dimensions(self.batchscale_sub_out, self.trainbatch.action[1])) then
      self.batchscale_sub_out =  self.trainbatch.action[1]:clone()
      for i = 1,self.action_bounds:size(2) do
        self.batchscale_sub_out[{{},{},{},i}]:fill((self.action_bounds[2][i] + self.action_bounds[1][i])/2) -- (max+min)/2
      end
      self.batchscale_mul_out = self.trainbatch.action[1]:clone()
      for i = 1,self.action_bounds:size(2) do
        self.batchscale_mul_out[{{},{},{},i}]:fill(2/(self.action_bounds[2][i] - self.action_bounds[1][i])) -- 2/(max-min)
      end
    end
    self.trainbatch.action[1]:add(-1,self.batchscale_sub_out)
    self.trainbatch.action[1]:cmul(self.batchscale_mul_out)
  end
	if not trainsettings.state_index then
		if #self.trainbatch.state == 1 then
			trainsettings.state_index = 1
		end
	end
  assert(trainsettings.state_index==1,"FIX TANHSCALE FOR MULTIPLE STATES")
	if ((not trainsettings.seq_properties) or trainsettings.seq_properties.length <=1) then		
		assert(trainsettings.state_index,"state_index not set and multiple states available")
		assert(trainsettings.batchupdates,"batchupdates not set (number of minibatch updates per train() call)")
		assert(trainsettings.gamma,"gamma RL parameter not set (DDPG uses TD learning)")
		trainsettings.minq = MINIMIZEQ

		for i=1,trainsettings.batchupdates do
			batch_index = math.max(1,i%self.trainbatch.state[1]:size(1))
			self:train_noseq(batch_index,trainsettings.state_index,trainsettings.gamma,trainsettings, experiencedb)
		end
	else
		assert(false, "DDPG not (yet) implemented for sequences")
	end
end

function ddpg:train_noseq(batch_index,state_index,gamma,train_settings, experiencedb)
	local sequence_index = 1
		
		-- make sure s, s' and a are scaled!
		function evaluateQ(paramx_)
			if paramx_ ~= self.train_networks.critic.paramx then self.train_networks.critic.paramx:copy(paramx_) end
			--FP		
		
			local s 	    	= self.trainbatch.state[state_index][{batch_index,sequence_index}]
			local a		    	= self.trainbatch.action[state_index][{batch_index,sequence_index}]
			local next_s  	= self.trainbatch.next_state[state_index][{batch_index,sequence_index}]	
			local r 	    	= self.trainbatch.reward[{batch_index,sequence_index}]
			local next_a  	= self.train_networks.FROZENactor.network:forward(next_s) 	
			if cutorch then 
				cutorch:synchronize()		
			end
			local Qtarget 	= self.train_networks.FROZENcritic.network:forward({next_s,next_a})
			if cutorch then 
				cutorch:synchronize()		
			end  
			local target   	= torch.add(r,gamma,Qtarget)	  	
      local Qpred 		= self.train_networks.critic.network:forward({s,a})
			if cutorch then 
				cutorch:synchronize()		
			end		
			--BP
			
			self.lastTDE 	= self.train_networks.criterion:forward(Qpred,target)
			if train_settings.logTDE then
				local TDEtens = torch.Tensor(target:size()):type(Qpred:type())
				TDEtens:add(Qpred,-1.0,target:clone():resizeAs(Qpred))
				experiencedb:update_extra_info("TDE",TDEtens:clone():double():abs(),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end
      if train_settings.logQPred then    
				experiencedb:update_extra_info("QPRED",Qpred:clone():double(),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end   
      if train_settings.logQFpred then    
				experiencedb:update_extra_info("QFPRED",Qtarget:clone():double(),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end
      if train_settings.logUPDATE then        
				experiencedb:update_extra_info("UPDATE",self.trainbatch.db_indices[{batch_index,sequence_index}]:clone():fill(self.updateCount),self.trainbatch.db_indices[{batch_index,sequence_index}])
			end
			self.train_networks.critic.paramdx:zero()
			if cutorch then 
				cutorch:synchronize()		
			end 		
			local errorgrad 		= self.train_networks.criterion:backward(Qpred,target)
			self.train_networks.critic.network:backward({s,a},errorgrad)
			if cutorch then 
				cutorch:synchronize()		
			end

			if self.settings.critic.knownQ and self.settings.critic.knownQ.use then
	  		if not(self.knownQ) then
	  			self.knownQ = {}
	  			self.knownQ.lastTDE = 0
	  			self.knownQ.state = self.settings.critic.knownQ.states:typeAs(self.train_networks.critic.paramx)
	  			self.knownQ.action = self.settings.critic.knownQ.actions:typeAs(self.train_networks.critic.paramx)
	  			self.knownQ.Q = self.settings.critic.knownQ.Qs:typeAs(self.train_networks.critic.paramx)
					---- SCALE KNOWNQs
					if (self.state_bounds) then 
			      local knownQ_sub_in =  self.knownQ.state:clone()
	          for i = 1,self.state_bounds:size(2) do
			        knownQ_sub_in[{{},i}]:fill((self.state_bounds[2][i] + self.state_bounds[1][i])/2) -- (max+min)/2
			      end
			      local knownQ_mul_in = self.knownQ.state:clone()
			      for i = 1,self.state_bounds:size(2) do
			        knownQ_mul_in[{{},i}]:fill(2/(self.state_bounds[2][i] - self.state_bounds[1][i])) -- 2/(max-min)
			      end
			    	self.knownQ.state:add(-1,knownQ_sub_in)
			    	self.knownQ.state:cmul(knownQ_mul_in)
			  	end	  
				  --TANHSCALE  KNOWNQ-------------------------------------------
				  if (self.action_bounds) then   
			      local QactionScaleSub =  self.knownQ.action:clone()
			      for i = 1,self.action_bounds:size(2) do
			        QactionScaleSub[{{},i}]:fill((self.action_bounds[2][i] + self.action_bounds[1][i])/2) -- (max+min)/2
			      end
			      local QactionScaleMul = self.knownQ.action:clone()
			      for i = 1,self.action_bounds:size(2) do
			        QactionScaleMul[{{},i}]:fill(2/(self.action_bounds[2][i] - self.action_bounds[1][i])) -- 2/(max-min)
			      end
				    self.knownQ.action:add(-1,QactionScaleSub)
				    self.knownQ.action:cmul(QactionScaleMul)
				  end
				end -- knownQ init
				--local counter = 1
        
				local Qpred 		= self.train_networks.critic.network:forward({self.knownQ.state,self.knownQ.action})
        if cutorch then 
        	cutorch:synchronize()		
        end	
			  self.knownQ.lastTDE 	= self.train_networks.criterion:forward(Qpred,self.knownQ.Q)
			  if cutorch then 
			  	cutorch:synchronize()		
			  end

				if (self.knownQ.lastTDE > self.lastTDE )--and counter < trainsettings.batchupdates) do
					then
				   -- Train with known Q untill TDE knownQ < avg TDE
				  local errorgrad 		= self.train_networks.criterion:backward(Qpred,self.knownQ.Q)
					self.train_networks.critic.network:backward({self.knownQ.state,self.knownQ.action},errorgrad)
					if cutorch then 
						cutorch:synchronize()		
					end
				end			
			end
			-- detect nan gradient
			if (self.train_networks.critic.paramdx:sum() ~= self.train_networks.critic.paramdx:sum()) then
				print("DDPG: NaN gradient in critic!")
				self.train_networks.critic.paramdx:zero()
			end
			-- l2 reg
			if self.trainsettings.critic_L2norm then
				self.train_networks.critic.paramdx:add( self.train_networks.critic.paramx:clone():mul( self.trainsettings.critic_L2norm ) )
			end
			return 0 , self.train_networks.critic.paramdx 
		end	

		function evaluateP(paramx_)
			if paramx_ ~= self.train_networks.actor.paramx then self.train_networks.actor.paramx:copy(paramx_) end
			--FP
			local s 	    	= self.trainbatch.state[state_index][{batch_index,sequence_index}]		
			local a 				= self.train_networks.actor.network:forward(s)
			self.train_networks.critic.paramdx:zero()
			self.train_networks.critic.network:forward({s,a})
			if cutorch then 
				cutorch:synchronize()		
			end
			-- BP
			local derivsign = -1
			if (train_settings.minq) then
				derivsign = 1
			end
			local ones 			= torch.Tensor(a:size(1),1):fill(derivsign):type(a:type())
			local dQda 			= self.train_networks.critic.network:backward({s,a},ones)[2]
			if cutorch then 
				cutorch:synchronize()		
			end
			self.train_networks.actor.paramdx:zero() 
			if cutorch then 
				cutorch:synchronize()		
			end
			self.train_networks.actor.network:backward(s,dQda)
			if cutorch then 
				cutorch:synchronize()		
			end				
			-- detect nan gradient
			if (self.train_networks.actor.paramdx:sum() ~= self.train_networks.actor.paramdx:sum()) then
				print("DDPG: NaN gradient in actor!")
				self.train_networks.actor.paramdx:zero()
			end
			if self.trainsettings.actor_L2norm then
				self.train_networks.actor.paramdx:add( self.train_networks.actor.paramx:clone():mul( self.trainsettings.actor_L2norm ) )
			end
			return 0,self.train_networks.actor.paramdx 
		end		
	
	-- Start of the training function -- 
	optim_settings = train_settings.optimsettings
	self.train_networks.critic.network:training()
	--self.train_networks.FROZENcritic.network:training()
	self.train_networks.actor.network:training()
	--self.train_networks.FROZENactor.network:training()  
	
	assert(optim_settings,"optim_settings not set")
	assert(optim_settings.optimfunction,"optimfunction not set")
	assert(optim_settings.configActor,"configActor not set")
	assert(optim_settings.configCritic,"configCritic not set")
	assert(optim_settings.targetlowpass,"targetlowpass not set (lowpass filter constant for updating the networks used for the target values.)") 

	optim_settings.optimfunction(evaluateQ, self.train_networks.critic.paramx, optim_settings.configCritic, self.optimStateCritic)
	optim_settings.optimfunction(evaluateP, self.train_networks.actor.paramx, optim_settings.configActor, self.optimStateActor)
	
	local QFP = self.train_networks.critic.paramx:clone()*optim_settings.targetlowpass + self.train_networks.FROZENcritic.paramx:clone()*(1-optim_settings.targetlowpass)
	self.train_networks.FROZENcritic.paramx:copy(QFP)
	local PFP = self.train_networks.actor.paramx:clone()*optim_settings.targetlowpass + self.train_networks.FROZENactor.paramx:clone()*(1-optim_settings.targetlowpass)
	self.train_networks.FROZENactor.paramx:copy(PFP)
	
	-- ensure the frozen networks are always on evaluate (but only after having gotten initial params)
	self.train_networks.FROZENactor.network:evaluate()
	self.train_networks.FROZENcritic.network:evaluate()	
	
end

function ddpg:getQ(state, action)
  state = state:resize(state:nElement()/self.settings.statesize,self.settings.statesize):typeAs(self.train_networks.critic.paramx)
  action = action:resize(action:nElement()/self.settings.actionsize,self.settings.actionsize):typeAs(self.train_networks.critic.paramx)
  self.train_networks.critic.network:evaluate()
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
  if (self.action_bounds) then 
	  if (not same_dimensions(self.ascale_sub, action)) then
	      self.ascale_sub =  action:clone()
	      for i = 1,self.action_bounds:size(2) do
	        self.ascale_sub[{{},i}]:fill((self.action_bounds[2][i] + self.action_bounds[1][i])/2) -- (max+min)/2
	      end
	      self.ascale_mul = action:clone()
	      for i = 1,self.action_bounds:size(2) do
	        self.ascale_mul[{{},i}]:fill(2/(self.action_bounds[2][i] - self.action_bounds[1][i])) -- 2/(max-min)
	      end
	  end
    
    action:add(-1,self.ascale_sub)
    action:cmul(self.ascale_mul)
  end
	return self.train_networks.critic.network:forward({state,action})
		--self.train_networks.critic.network:evaluate()		
end	


function ddpg:valQTDE(state, next_state, action, reward)
  state = state:resize(state:nElement()/self.settings.statesize,self.settings.statesize):typeAs(self.train_networks.critic.paramx)
  action = action:resize(action:nElement()/self.settings.actionsize,self.settings.actionsize):typeAs(self.train_networks.critic.paramx)
  next_state = next_state:resize(next_state:nElement()/self.settings.statesize,self.settings.statesize):typeAs(self.train_networks.critic.paramx)
  action = action:resize(action:nElement()/self.settings.actionsize,self.settings.actionsize):typeAs(self.train_networks.critic.paramx)
  self.train_networks.critic.network:evaluate()
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

  if (self.action_bounds) then 
	  if (not same_dimensions(self.ascale_sub, action)) then
	      self.ascale_sub =  action:clone()
	      for i = 1,self.action_bounds:size(2) do
	        self.ascale_sub[{{},i}]:fill((self.action_bounds[2][i] + self.action_bounds[1][i])/2) -- (max+min)/2
	      end
	      self.ascale_mul = action:clone()
	      for i = 1,self.action_bounds:size(2) do
	        self.ascale_mul[{{},i}]:fill(2/(self.action_bounds[2][i] - self.action_bounds[1][i])) -- 2/(max-min)
	      end
	  end
    
    action:add(-1,self.ascale_sub)
    action:cmul(self.ascale_mul)
  end
	local next_action = self.train_networks.actor.network:forward(state)
	

	local Qs =  self.train_networks.critic.network:forward({state,action})
	


		--self.train_networks.critic.network:evaluate()		
end	
	

