require "drl_dqn"

local dqnr,parent = torch.class('drl_dqnr','drl_dqn')

function dqnr:__init(controller,trainer,settings)
	parent.__init(self,controller,trainer,settings)
end

function dqnr:create_sumnet()
  local q_currentstate			=	nn.Identity()()	
	local q_max_nextstate			= nn.Identity()()	
	local next_state_weight		= nn.Identity()()	-- -1* (gamma * (1-terminal))
	
	local forgetandterminate 	= nn.CMulTable()({q_max_nextstate	,next_state_weight})
	local rewardestimate			= nn.CSubTable()({q_currentstate,forgetandterminate})

	local network 							= nn.gModule({q_currentstate, q_max_nextstate, next_state_weight},{rewardestimate})
	return network
end

function dqnr:create_train_networks()
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
	self.train_networks.Q.second_network			= self.train_networks.Q.network:clone('weight','bias','gradWeight','gradBias')
	self.train_networks.FROZENQ.paramx 				= FQ_paramx
	self.train_networks.FROZENQ.paramdx 			= FQ_paramdx
	self.train_networks.bellmansum						= self:create_sumnet()

	self.optimStateQ 													= {}
end

function dqnr:train_noseq(batch_index,state_index,gamma,train_settings, experiencedb)
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
			local gammaterm 			
			if self.trainbatch.terminal then
				gammaterm = self.trainbatch.terminal[{batch_index,sequence_index}]:clone()
			else
				gammaterm = r:clone():zero()
			end
			-- to use for the Q(s') in a multiplicative way (becomes gamma for non terminal next states amd 0 for terminal ones)
			gammaterm:mul(-gamma):add(gamma)
			local maxqnexts, policy = self.train_networks.FROZENQ.network:forward(next_s):max(2)
			
      local Qpreds 			= self.train_networks.Q.network:forward(s)
			local Qpredsnexts = self.train_networks.Q.second_network:forward(next_s)
			local q = torch.Tensor(Qpreds:size(1)):typeAs(Qpreds)	
			local nextq = torch.Tensor(Qpreds:size(1)):typeAs(Qpreds)	
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
					q[i] 			= Qpreds[i][actionbatchindex[i]]
					nextq[i] 	= Qpredsnexts[i][policy[i][1]] 
			end
			if cutorch then 
				cutorch:synchronize()		
			end		
			local predictedrewards = self.train_networks.bellmansum:forward({q,nextq,gammaterm})
			if cutorch then 
				cutorch:synchronize()		
			end

			--BP
			
			self.lastTDE 	= self.train_networks.criterion:forward(predictedrewards,r)
			if train_settings.logTDE then
				--calculate temporal difference error per database sample
				local TDEtens = torch.Tensor(r:size()):type(r:type())
				TDEtens:add(predictedrewards,-1.0,r:clone():resizeAs(predictedrewards))
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
			local errorgrad 		= self.train_networks.criterion:backward(predictedrewards,r)
			local dq1, dq2, dgt = unpack(self.train_networks.bellmansum:backward({q,nextq,gammaterm},errorgrad))

			local Qgradient1 		= torch.Tensor(Qpreds:size()):typeAs(Qpreds):zero()
			local Qgradient2 		= torch.Tensor(Qpreds:size()):typeAs(Qpreds):zero()
			
			for i = 1,a:size(1) do
				Qgradient1[i][actionbatchindex[i]] 	= dq1[i]
				Qgradient2[i][policy[i][1]] 						= dq2[i]
			end	

			self.train_networks.Q.network:backward(s,Qgradient1)
			self.train_networks.Q.second_network:backward(next_s,Qgradient2)
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
	
	if (self.gradientupdatecount % optim_settings.freezecount == 0) then
		self.train_networks.FROZENQ.paramx:copy(self.train_networks.Q.paramx)
	end
	-- ensure the frozen networks are always on evaluate (but only after having gotten initial params)
	self.train_networks.FROZENQ.network:evaluate()
end

