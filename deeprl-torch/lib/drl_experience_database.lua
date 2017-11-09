--[[

		Tim de Bruin 2015
		deepRL-torch
		experience database class.
		Constructs the full RL state from its parts and / or
		maintains the experience database for training,

--]]
if USEGPU then
	require "cunn"
end

local em = torch.class('experience_memory')

function em:__init(short_term_memory,long_term_memory,settings)
	self.expm_settings 							= settings
	self.get_OSAR_function 					= settings.get_OSAR_function
	self.send_OSAR_function 				= settings.send_OSAR_function
	if short_term_memory then -- for full RL state reconstruction from partial info
		self.short_term_memory_size 	= settings.short_term_memory_size or 10000
		self.full_state_parts 			  = {} -- the partial state objects including memory and functions
    self.current_full_state       = {} -- the last full state tensor table
    for i = 1,#settings.full_state_dimension do -- number of full states (1 unless multi-modal non concatinated input)
      table.insert(self.full_state_parts,{})
      table.insert(self.current_full_state, torch.Tensor(settings.full_state_dimension[i]):zero())
    end
    if (settings.observation_dimension) then
		  self.observation_parts 			 	= {} -- the observation objects including memory and functions
		  self.current_observation 			= {}
		  for i = 1,#settings.observation_dimension do -- number of observation tensors (1 unless multi-modal non concatinated input)
		    table.insert(self.current_observation, torch.Tensor(settings.observation_dimension[i]):zero())
		  end
		end
	end

	if long_term_memory then
		self.state_to_add = { current = {}, previous = {}} -- remember state action and reward of the last time step
		assert(settings.experience_replay_size,"settings.experience_replay_size not set")
		self.experience_replay_size 	= settings.experience_replay_size -- number of experiences
		assert(settings.RL_state_parts,"settings.RL_state_parts not specified, specify the components of the RL state (state action reward next_state next_action observation")
		self.experience_database = {current_write_index = 1, previous_write_index = 1, last_write_index = 0}
		self.experience_database.time_indices = torch.Tensor(self.experience_replay_size)
		self.experience_database.db_indices = torch.linspace(1,self.experience_replay_size,self.experience_replay_size)
		self.experience_database.sequence_indices = torch.Tensor(self.experience_replay_size)
		self.last_sequence_index = 0
		if settings.extra_info then
			self.experience_database.extra_info = {}
			assert(type(settings.extra_info=="table"))
			for i,state in ipairs(settings.extra_info) do
				assert(state.name and type(state.name=="String"),"No name given to extra DB information state")
				assert(state.default_value,"no default value given")
				assert(type(state.default_value)=="number","Only numbers supported for now.")
				table.insert(self.experience_database.extra_info , {name = state.name, values = torch.Tensor(self.experience_replay_size):fill(state.default_value), default_value = state.default_value})
			end
		end


		-- TODO: maybe store every seperate observation/state in 1D and resize() at the last moment?
		if settings.RL_state_parts.state then
			self.experience_database.state = {}
			for i = 1,#settings.full_state_dimension do
				local size 	= torch.LongStorage(#settings.full_state_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = settings.full_state_dimension[i][j-1]
				end
				table.insert(self.experience_database.state, torch.Tensor(size):zero())
			end
		end
		if settings.RL_state_parts.next_state then
			self.experience_database.next_state = {}
			for i = 1,#settings.full_state_dimension do
				local size 	= torch.LongStorage(#settings.full_state_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = settings.full_state_dimension[i][j-1]
				end
				table.insert(self.experience_database.next_state, torch.Tensor(size):zero())
			end
		end
		if settings.RL_state_parts.action then
			self.experience_database.action = {}
			for i = 1,#settings.action_dimension do
				local size 	= torch.LongStorage(#settings.action_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = settings.action_dimension[i][j-1]
				end
				table.insert(self.experience_database.action, torch.Tensor(size):zero())
			end
		end
		if settings.RL_state_parts.next_action then
			self.experience_database.next_action = {}
			for i = 1,#settings.action_dimension do
				local size 	= torch.LongStorage(#settings.action_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = settings.action_dimension[i][j-1]
				end
				table.insert(self.experience_database.next_action, torch.Tensor(size):zero())
			end
		end
		if settings.RL_state_parts.reward then
			self.experience_database.reward = torch.Tensor(self.experience_replay_size)
		end
		if settings.RL_state_parts.terminal then
			self.experience_database.terminal = torch.Tensor(self.experience_replay_size)
		end
		if settings.RL_state_parts.observation then
				self.experience_database.observation = {}
				for i = 1,#settings.observation_dimension do
					local size 	= torch.LongStorage(#settings.observation_dimension[i]+1)
					size[1] 	= self.experience_replay_size
					for j = 2,#size do
						size[j] = settings.observation_dimension[i][j-1]
					end
					table.insert(self.experience_database.observation, torch.Tensor(size):zero())
				end
		end
	end

end

function em:reset()
	if self.experience_database then

		if self.experience_database.extra_info then
			for i,state in ipairs(self.experience_database.extra_info) do
				state.values:fill(state.default_value)
			end
		end

		self.state_to_add = { current = {}, previous = {}} -- remember state action and reward of the last time step
		self.experience_database.current_write_index = 1
		self.experience_database.last_write_index = 0
		self.experience_database.time_indices:fill(-1)
		self.experience_database.sequence_indices:fill(-1)
		if self.expm_settings.RL_state_parts.state then
			self.experience_database.state = {}
			for i = 1,#self.expm_settings.full_state_dimension do
				local size 	= torch.LongStorage(#self.expm_settings.full_state_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = self.expm_settings.full_state_dimension[i][j-1]
				end
				table.insert(self.experience_database.state, torch.Tensor(size):zero())
			end
		end
		if self.expm_settings.RL_state_parts.next_state then
			self.experience_database.next_state = {}
			for i = 1,#self.expm_settings.full_state_dimension do
				local size 	= torch.LongStorage(#self.expm_settings.full_state_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = self.expm_settings.full_state_dimension[i][j-1]
				end
				table.insert(self.experience_database.next_state, torch.Tensor(size):zero())
			end
		end
		if self.expm_settings.RL_state_parts.action then
			self.experience_database.action = {}
			for i = 1,#self.expm_settings.action_dimension do
				local size 	= torch.LongStorage(#self.expm_settings.action_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = self.expm_settings.action_dimension[i][j-1]
				end
				table.insert(self.experience_database.action, torch.Tensor(size):zero())
			end
		end
		if self.expm_settings.RL_state_parts.next_action then
			self.experience_database.next_action = {}
			for i = 1,#self.expm_settings.action_dimension do
				local size 	= torch.LongStorage(#self.expm_settings.action_dimension[i]+1)
				size[1] 	= self.experience_replay_size
				for j = 2,#size do
					size[j] = self.expm_settings.action_dimension[i][j-1]
				end
				table.insert(self.experience_database.next_action, torch.Tensor(size):zero())
			end
		end
		if self.expm_settings.RL_state_parts.reward then
			self.experience_database.reward = torch.Tensor(self.experience_replay_size):zero()
		end
		if self.expm_settings.RL_state_parts.terminal then
			self.experience_database.terminal = torch.Tensor(self.experience_replay_size):zero()
		end
		if self.expm_settings.RL_state_parts.observation then
				self.experience_database.observation = {}
				for i = 1,#self.expm_settings.observation_dimension do
					local size 	= torch.LongStorage(#self.expm_settings.observation_dimension[i]+1)
					size[1] 	= self.experience_replay_size
					for j = 2,#size do
						size[j] = self.expm_settings.observation_dimension[i][j-1]
					end
					table.insert(self.experience_database.observation, torch.Tensor(size):zero())
				end
		end
	end
collectgarbage()
end

function em:reset_getfunctions()
	self.full_state_parts = {}
	self.observation_parts = {}
	print("FIX THIS! (experiencedb:reset_getfunctions needs settings")
  --[[for i = 1,#settings.full_state_dimension do -- number of full states (1 unless multi-modal non concatinated input)
    table.insert(self.full_state_parts,{})
    table.insert(self.current_full_state, torch.Tensor(settings.full_state_dimension[i]):zero())
  end
  if (settings.observation_dimension) then
	  self.observation_parts 			 	= {} -- the observation objects including memory and functions
	  self.current_observation 			= {}
	  for i = 1,#settings.observation_dimension do -- number of observation tensors (1 unless multi-modal non concatinated input)
	    table.insert(self.current_observation, torch.Tensor(settings.observation_dimension[i]):zero())
	  end
	end
	]]
	self.full_state_parts[1] = {}
end

-- each full state can be made up out of several state parts
function em:addPartialState(full_state_index,partial_state)
    table.insert(self.full_state_parts[full_state_index],partial_state)
end

-- each observation is independent
function em:addObservation(observation)
    table.insert(self.observation_parts,observation)
end

function em:setAction(action, extra_information)
		self.action_state = action
		if extra_information then
			-- deal with it, probably in a child class
		end
end

function em:setReward(reward, extra_information)
		self.reward_state = reward
		self.cost 				= extra_information.cost
		if extra_information then
			-- deal with it, probably in a child class
			self.reward_delay = extra_information.delay
		else
			self.reward_delay = false
		end
end

function em:setTerminal(terminal, extra_information)
		self.terminal_state = terminal
		if extra_information then
			-- deal with it, probably in a child class
		end
end

function em:construct_full_state(time_index,sequence_index)
  -- update the partial states
  for idx1, full_state_part in ipairs(self.full_state_parts) do
    for indx2, partial_state in ipairs(full_state_part) do
      partial_state:receive_for_time(time_index,sequence_index)
    end
  end
  -- construct the full state
  for fsi,full_state_part in ipairs(self.full_state_parts) do
    self.current_full_state[fsi]:zero()
    local index = 1
    for idx1, partial_state in ipairs(full_state_part) do
      for i = 0,partial_state.prevPerFullState do
        local state_part    = partial_state:get_value_for_timestep(time_index - i,sequence_index)

        local state_length  = state_part:size(1)
        self.current_full_state[fsi][{{index,index+state_length-1}}] = state_part
        index = index + state_length
      end
    end
  end
	return self.current_full_state
end

function em:collect_observations(time_index,sequence_index)
  -- update the observations states
  assert(time_index and sequence_index,"provide the time and sequence indices")
  for idx1, observation in ipairs(self.observation_parts) do
      observation:receive_for_time(time_index,sequence_index)
      self.current_observation[idx1] = observation:get_value_for_timestep(time_index,sequence_index)
  end
	return self.current_observation
end


-- collect the observations state action and reward by interacting with the environment
function em:collect_OSAR(time_index,sequence_index)
	local state 					= self:construct_full_state(time_index,sequence_index)
	local action 					= self.action_state:get_value_for_timestep(time_index,sequence_index,state)
	local reward 					= self.reward_state:get_value_for_timestep(time_index,sequence_index)
	local terminal  			= 0
	if self.terminal_state then terminal = self.terminal_state:get_value_for_timestep(time_index,sequence_index)
	end
	local observation     = false
  if(self.expm_settings.RL_state_parts.observation) then
		observation 	      = self:collect_observations(time_index,sequence_index)
	end
	self.OSAR = {time_index = time_index, sequence_index = sequence_index, state = state,action = action,reward = reward, terminal = terminal, observation = observation}
end

-- collect observations state action and reward by receiving them from another module / loading
function em:receive_OSAR(time_index)
	-- receive function
  self:get_OSAR_function()
end


-- send the observations state action and reward to another module
function em:send_OSAR(time_index)
	self:send_OSAR_function(time_index)
end

function em:get_OSAR_copy()
	local osarcopy = {}
	assert(self.OSAR ,"No OSAR to set")

	osarcopy.time_index  = self.OSAR.time_index
	osarcopy.sequence_index = self.OSAR.sequence_index
	osarcopy.state  = {}
	for i = 1,#self.OSAR.state do
		osarcopy.state[i] = self.OSAR.state[i]:clone()
	end
	osarcopy.action      	= self.OSAR.action:clone()
	if torch.isTensor(self.OSAR.reward) then
		osarcopy.reward      	= self.OSAR.reward:clone()
	else
		osarcopy.reward      	= self.OSAR.reward
	end
	if torch.isTensor(self.OSAR.terminal) then
		osarcopy.terminal      	= self.OSAR.terminal:clone()
	else
		osarcopy.terminal      	= self.OSAR.terminal
	end
	if torch.isTensor(self.OSAR.observation) then
		osarcopy.observation      	= self.OSAR.observation:clone()
	else
		osarcopy.observation      	= self.OSAR.observation
	end
	return osarcopy
end

--- Write the experience to the database
function em:add_RL_state_to_db()
  assert(self.OSAR ,"No OSAR to set")

  local time_index  = self.OSAR.time_index
  local sequence_index = self.OSAR.sequence_index
  self.last_sequence_index = sequence_index
  local full_state  = {}
  for i = 1,#self.OSAR.state do
  	full_state[i] = self.OSAR.state[i]:clone()
  end
  local action      = self.OSAR.action:clone()
  local reward      = self.OSAR.reward
  local terminal 		= self.OSAR.terminal
  local observation = self.OSAR.observation

    local write_to_db = function(db,index,object)
      if(torch.type(object)=="table")then
          for idx,fsp in ipairs(object) do
            db[idx][index]:copy(fsp)
          end
      else
        db[1][index]:copy(object)
      end
    end

  if (time_index and sequence_index and full_state and action and reward) then
		if self.state_to_add.current.time_index and self.state_to_add.current.time_index == time_index - 1 and self.state_to_add.current.sequence_index == sequence_index then
		  -- prev state = current state, current state = new state, save to DB
		  self.state_to_add.previous.time_index 			= self.state_to_add.current.time_index
		  self.state_to_add.previous.sequence_index 	= self.state_to_add.current.sequence_index
		  self.state_to_add.previous.state 						= self.state_to_add.current.state
		  self.state_to_add.previous.action 					= self.state_to_add.current.action
		  self.state_to_add.previous.reward 					= self.state_to_add.current.reward
		  self.state_to_add.previous.terminal					= self.state_to_add.current.terminal
		  self.state_to_add.previous.observation 			= self.state_to_add.current.observation

		  self.state_to_add.current.state   					= full_state
		  self.state_to_add.current.action  					= action
		  self.state_to_add.current.reward  					= reward
		  self.state_to_add.current.terminal  				= terminal
		  self.state_to_add.current.observation 			= observation
		  self.state_to_add.current.time_index 				=	time_index
		  self.state_to_add.current.sequence_index 		= sequence_index

		  -- write to the DB
		  self.experience_database.time_indices[self.experience_database.current_write_index] = self.state_to_add.previous.time_index

		  self.experience_database.sequence_indices[self.experience_database.current_write_index] = self.state_to_add.previous.sequence_index

			if self.experience_database.extra_info then
				for i,state in ipairs(self.experience_database.extra_info) do
					state.values[self.experience_database.current_write_index] = (state.default_value)
				end
			end

		  if self.experience_database.state then
		    write_to_db(self.experience_database.state,self.experience_database.current_write_index,self.state_to_add.previous.state)
		  end
		  if self.experience_database.next_state then
		    write_to_db(self.experience_database.next_state,self.experience_database.current_write_index,self.state_to_add.current.state)
		  end
		  if self.experience_database.action then
		    write_to_db(self.experience_database.action,self.experience_database.current_write_index,self.state_to_add.previous.action)
		  end
		  if self.experience_database.next_action then
		    write_to_db(self.experience_database.next_action,self.experience_database.current_write_index,self.state_to_add.current.action)
		  end
		  if self.experience_database.reward then
				if self.reward_delay then
					self.experience_database.reward[self.experience_database.current_write_index] = reward
				else
					self.experience_database.reward[self.experience_database.current_write_index] = self.state_to_add.previous.reward
				end
			end
			if self.experience_database.terminal then
			  self.experience_database.terminal[self.experience_database.current_write_index] = terminal
			end
			if self.experience_database.observation then
		    write_to_db(self.experience_database.observation,self.experience_database.current_write_index,self.state_to_add.previous.observation)
		  end

		  -- advance DB write index
		  local writenidx = self.experience_database.current_write_index
		  if(self.experience_database.current_write_index > self.experience_database.last_write_index) then
		    self.experience_database.last_write_index = self.experience_database.current_write_index
		  end

		  self:advance_db_write_index()
		  return writenidx
		else -- no previous experience
		  -- current state = new state
		  self.state_to_add.current.state   					= full_state
		  self.state_to_add.current.action 	 					= action
		  self.state_to_add.current.reward  					= reward
		  self.state_to_add.current.observation 			= observation
		  self.state_to_add.current.time_index				= time_index
		  self.state_to_add.current.sequence_index 		= sequence_index
		end
	else
		print("WARNING NO EXPERIENCE TO ADD TO THE DATABASE")
		if not (time_index) then print("time_index not set") end
		if not (sequence_index) then print("sequence_index not set") end
		if not (full_state) then print("full_state not set") end
		if not (action) then print ("action not set") end
		if not (reward) then print ("reward not set") end
	end
end

--- DUMMY - same side effects as add state, without adding to the db.
function em:dont_add_RL_state_to_db()
  assert(self.OSAR ,"No OSAR to set")

  local time_index  = self.OSAR.time_index
  local sequence_index = self.OSAR.sequence_index
  self.last_sequence_index = sequence_index
  local full_state  = {}
  for i = 1,#self.OSAR.state do
  	full_state[i] = self.OSAR.state[i]:clone()
  end
  local action      = self.OSAR.action:clone()
  local reward      = self.OSAR.reward
  local terminal 		= self.OSAR.terminal
  local observation = self.OSAR.observation

  if (time_index and sequence_index and full_state and action and reward) then
		if self.state_to_add.current.time_index and self.state_to_add.current.time_index == time_index - 1 and self.state_to_add.current.sequence_index == sequence_index then
		  -- prev state = current state, current state = new state, save to DB
		  self.state_to_add.previous.time_index 			= self.state_to_add.current.time_index
		  self.state_to_add.previous.sequence_index 	= self.state_to_add.current.sequence_index
		  self.state_to_add.previous.state 						= self.state_to_add.current.state
		  self.state_to_add.previous.action 					= self.state_to_add.current.action
		  self.state_to_add.previous.reward 					= self.state_to_add.current.reward
		  self.state_to_add.previous.terminal					= self.state_to_add.current.terminal
		  self.state_to_add.previous.observation 			= self.state_to_add.current.observation

		  self.state_to_add.current.state   					= full_state
		  self.state_to_add.current.action  					= action
		  self.state_to_add.current.reward  					= reward
		  self.state_to_add.current.terminal  				= terminal
		  self.state_to_add.current.observation 			= observation
		  self.state_to_add.current.time_index 				=	time_index
		  self.state_to_add.current.sequence_index 		= sequence_index

		  -- write to the DB
		  -- deleted

		  -- advance DB write index
		  local writenidx = self.experience_database.current_write_index
		  if(self.experience_database.current_write_index > self.experience_database.last_write_index) then
		    self.experience_database.last_write_index = self.experience_database.current_write_index
		  end

		  self:advance_db_write_index()
		  return writenidx
		else -- no previous experience
		  -- current state = new state
		  self.state_to_add.current.state   					= full_state
		  self.state_to_add.current.action 	 					= action
		  self.state_to_add.current.reward  					= reward
		  self.state_to_add.current.observation 			= observation
		  self.state_to_add.current.time_index				= time_index
		  self.state_to_add.current.sequence_index 		= sequence_index
		end
	else
		print("WARNING NO EXPERIENCE TO ADD TO THE DATABASE")
		if not (time_index) then print("time_index not set") end
		if not (sequence_index) then print("sequence_index not set") end
		if not (full_state) then print("full_state not set") end
		if not (action) then print ("action not set") end
		if not (reward) then print ("reward not set") end
	end
end

--- WARNING!
--- This function should not be used in general! Use the one above instead.
function em:direct_RL_state_insert(rlstate)
	local write_to_db = function(db,index,object)
	  if(torch.type(object)=="table")then
	      for idx,fsp in ipairs(object) do
	        db[idx][index]:copy(fsp)
	      end
	  else
	    db[1][index]:copy(object)
	  end
	end

  -- write to the DB
  self.experience_database.time_indices[self.experience_database.current_write_index] = rlstate.time_index
  self.experience_database.sequence_indices[self.experience_database.current_write_index] = rlstate.sequence_index

	if self.experience_database.extra_info then
		for i,state in ipairs(self.experience_database.extra_info) do
			state.values[self.experience_database.current_write_index] = (state.default_value)
		end
	end

  if self.experience_database.state then
    write_to_db(self.experience_database.state,self.experience_database.current_write_index,rlstate.state)
  end
  if self.experience_database.next_state then
    write_to_db(self.experience_database.next_state,self.experience_database.current_write_index,rlstate.next_state)
  end
  if self.experience_database.action then
    write_to_db(self.experience_database.action,self.experience_database.current_write_index,rlstate.action)
  end
  if self.experience_database.next_action then
    write_to_db(self.experience_database.next_action,self.experience_database.current_write_index,rlstate.current.action)
  end
  if self.experience_database.reward then
			self.experience_database.reward[self.experience_database.current_write_index] = rlstate.reward
	end
	if self.experience_database.terminal then
	  self.experience_database.terminal[self.experience_database.current_write_index] = rlstate.terminal
	end
	if self.experience_database.observation then
    write_to_db(self.experience_database.observation,self.experience_database.current_write_index,rlstate.observation)
  end

  -- advance DB write index

  if(self.experience_database.current_write_index > self.experience_database.last_write_index) then
    self.experience_database.last_write_index = self.experience_database.current_write_index
  end
  self:advance_db_write_index()
end


function em:advance_db_write_index()
	self.experience_database.previous_write_index = self.experience_database.current_write_index
	if self.expm_settings.overwrite_policy == 'FIFO' then
		self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		if self.experience_database.current_write_index > self.experience_replay_size then
		  self.experience_database.current_write_index = 1
		end
	elseif self.expm_settings.overwrite_policy == 'HALF' then
		self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		if self.experience_database.current_write_index > self.experience_replay_size then
		  self.experience_database.current_write_index = math.floor(self.experience_database.last_write_index/2)
		end
	elseif self.expm_settings.overwrite_policy == 'TDE' then
		if self.experience_database.last_write_index < self.experience_replay_size then
			self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		else
			-- find the sample with the lowest TDE
				local info_state = false
				for i,state in ipairs(self.experience_database.extra_info) do
					if (state.name == 'TDE') then
						info_state = state
					end
				end
				assert(info_state, "TDE state not found in the extra_info database")

			local currentLowestTDE 	= math.huge
			local currentIndex 			= 1
			for index = 1,self.experience_replay_size do
				if (info_state.values[index] < currentLowestTDE) then
					currentLowestTDE 		= info_state.values[index]
					currentIndex 				= index
				end
			end
			self.experience_database.current_write_index = currentIndex
		end
	elseif self.expm_settings.overwrite_policy == 'DISTRIBUTION' then
		if self.experience_database.last_write_index < self.experience_replay_size then
			self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		else
			if not(self.lastDistributionRefresh) or self.lastDistributionRefresh < self.last_sequence_index then
				self.lastDistributionRefresh = self.last_sequence_index
				self.updateDistributions()
			end
      		self.experience_database.current_write_index = self.distribution_index()
		end
	elseif self.expm_settings.overwrite_policy == 'DISTRIBUTION_FAST' then
		if self.experience_database.last_write_index < self.experience_replay_size then
			self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		else
			if not(self.lastDistributionRefresh) or self.lastDistributionRefresh < self.last_sequence_index then
				self.lastDistributionRefresh = self.last_sequence_index
				self.updateDistributions()
			end
  		self.experience_database.current_write_index = self.distribution_index()
		end
	elseif self.expm_settings.overwrite_policy == 'OFFPOL' then
		if self.experience_database.last_write_index < self.experience_replay_size then
			self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		else
			if not(self.lastDistributionRefresh) or self.lastDistributionRefresh < self.last_sequence_index then
				self.lastDistributionRefresh = self.last_sequence_index
				self.updateDistributions()
			end
  		self.experience_database.current_write_index = self.distribution_index()
		end
	elseif self.expm_settings.overwrite_policy == 'RESERVOIR' then
		if self.experience_database.last_write_index < self.experience_replay_size then
			self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		else
  		-- cant simply drop the experience, so write it to a random location and have it overwritten (in runsim) if need be
  		self.experience_database.current_write_index = math.random(self.experience_replay_size)
		end
	elseif self.expm_settings.overwrite_policy == 'STOCHRANK' then
		if self.experience_database.last_write_index < self.experience_replay_size then
			self.experience_database.current_write_index = self.experience_database.current_write_index + 1
		else
			if not(self.lastDistributionRefresh) or self.lastDistributionRefresh < self.last_sequence_index then
				if not(self.overwrite_policy_buckets) then
					local nrbuckets = opt.seqlength*opt.samplefreq--math.max(2,math.min(math.ceil((self.experience_replay_size)/200),100))
					--
					local temp_indices = torch.linspace(1,self.experience_replay_size,self.experience_replay_size)
					local sampling_probabilities = torch.cumsum(torch.ones(self.experience_replay_size):cdiv(temp_indices):pow(self.expm_settings.overwrite_alpha))
						-- creating <batch_size> buckets of equal probability and sampling one experience uniformly from each
					self.overwrite_policy_buckets = torch.Tensor(1+nrbuckets)
					self.overwrite_policy_buckets[1] = 0
					for bb = 1,nrbuckets do
						local avi = temp_indices[sampling_probabilities:ge(sampling_probabilities[-1]*bb/(nrbuckets))]
						local first_index
						if avi:nElement() > 0 then
							first_index = avi:min()
						else
							first_index = 0
						end
						self.overwrite_policy_buckets[bb+1] = math.max(self.overwrite_policy_buckets[bb]+1,
							first_index)
					end
				end
				-- higher values are more desirable and will have >= probability of being kept in memory
				local desvals = self:get_extra_info(self.expm_settings.overwrite_metric)
				local sorted_dvals
				sorted_dvals, self.sorted_overwrite_indices = torch.sort(desvals) -- sorted from lowest to highest desirability
				self.lastDistributionRefresh = self.last_sequence_index
				self.currentBucket = 0
			end
			self.currentBucket = self.currentBucket + 1
			if self.currentBucket > (self.overwrite_policy_buckets:nElement() - 1) then
				self.currentBucket = 1
			end
			local index = math.random(self.overwrite_policy_buckets[self.currentBucket]+1,self.overwrite_policy_buckets[self.currentBucket+1])
			self.experience_database.current_write_index = self.sorted_overwrite_indices[index]
		end

	else
		assert(false,"Unknown overwrite_policy")
	end
end

function em:rewind_overwrite_index()
	self.experience_database.current_write_index = self.experience_database.previous_write_index
end


-- TODO! return only as many batches as necc

-- if seq_properties is set return data of size: <=batch_size x <=seq_properties.seq_length x data dimensions where the data is sequential in the seq_length dimension.  If seq_properties is not set then return data of size: <=batch_size x data dimensions with the data order randomized in the batch_size dimension.
--{state index}[batch number, sequence index, batch index, state dimension]
--trainvaltest = nil, "train" , "validate" , "test"
-- settings =  {batch_size,GPU,requested_parts,seq_properties,one_batch,trainvaltest,surprises, prioritized}
function em:get_mini_batch(batch_settings)
	local reqp 						= batch_settings.requested_parts or {observation = false, state = true, action = true, next_state = true, reward = true, terminal = true}
	local batch_size 			= batch_settings.batch_size
	local maxbatches  		= batch_settings.maxbatches or math.huge
	local GPU 						= batch_settings.GPU or false
	local seq_properties 	= batch_settings.seq_properties or {length = 1}
	local one_batch 			= batch_settings.one_batch or false
	-- rank based stochatsic prioritized experience replay with optional weighted importance sampling ( https://arxiv.org/pdf/1511.05952v4.pdf )
	local prioritized				= batch_settings.prioritized
	local prioritized_alpha = batch_settings.prioritized_alpha
	local prioritized_beta  = batch_settings.prioritized_beta or 0
	local countbasedis 			= batch_settings.countbasedimpsamp

	if countbasedis and not(self.bintab) then -- defined in runsim
		assert(false)
	end

	local function indexedcopy(object,indexes,returntensortype)
		local function tensorcopy(originalTensor,indexes,returntensortype)
			local returnTensor 	= originalTensor:index(1,indexes) -- copy with randomized order
			local dimensions 		= returnTensor:size():totable()
			assert(batch_size,"Batch size is nil")
			assert(seq_properties.length,"Sequence length is nil (give 1 when not using sequences)")
			dimensions[1] 			= batch_size
			table.insert(dimensions,1,seq_properties.length)
			table.insert(dimensions,1,-1)
			returnTensor 				= returnTensor:view(torch.LongStorage(dimensions))

			return returnTensor:type(returntensortype)
		end

		if object then
			if type(object)=="table" then
				local returntable = {}
				for i,j in ipairs(object) do
					returntable[i] = tensorcopy(j,indexes,returntensortype)
				end
				return returntable
			else
				return tensorcopy(object,indexes,returntensortype)
			end
		else
			return nil
		end
	end

	local db = self.experience_database
	local requestedTensorType = torch.getdefaulttensortype()
	local batch = {}
	if GPU then
		requestedTensorType = "torch.CudaTensor"
	end
	if not seq_properties then
		seq_properties = {}
	end
	if not seq_properties.length then
		seq_properties.length = 1
	end
	if seq_properties.length > 1 then
		assert(not(prioritized), "prioritized experience replay not yet implemented for sequences.")
		local startindices = {}
		local endindices = {}
		local count, seqId, lastTI, startidx = 0,0,0,0
		for i=1,db.last_write_index do
			if db.time_indices[i] == (lastTI + 1) and db.sequence_indices[i] == seqId then
				count 	= count + 1
				if count == seq_properties.length then
					table.insert(startindices,startidx)
					table.insert(endindices,i)
					seqId = -2
				end
			else
				count = 1
				seqId = db.sequence_indices[i]
				startidx = i
			end
			lastTI 	= db.time_indices[i]
		end
		local nrsequences = #startindices
		randomseqorder = torch.randperm(nrsequences)
		if nrsequences > 1 then
			local batches = 1
			if not(one_batch) then batches = math.min(math.max(1,math.floor(nrsequences/batch_size)),maxbatches) end
			local truebatch_size = batch_size
			batch_size = seq_properties.length
			batch_size = truebatch_size -- !
			local batchIndices = 	torch.LongTensor(batches*truebatch_size* seq_properties.length)
			local si, dbi = 1, startindices[randomseqorder[1]]
			for i=1,batchIndices:nElement() do
				if dbi > endindices[randomseqorder[si]] then
					si = si + 1
					if si > #startindices then
						si = 1
					end
					dbi = startindices[randomseqorder[si]]
				end
				batchIndices[i] = dbi
				dbi = dbi + 1
			end
			batchIndices:resize(batches,batch_size,seq_properties.length)
			batchIndices = batchIndices:transpose(2,3)
			batchIndices = batchIndices:reshape(batchIndices:nElement())
			if reqp.observation then batch.observation 	= indexedcopy(db.observation,batchIndices,requestedTensorType) end
			if reqp.state then batch.state 							= indexedcopy(db.state,batchIndices,requestedTensorType) end
			if reqp.action then	batch.action						= indexedcopy(db.action,batchIndices,requestedTensorType) end
			if reqp.next_state then batch.next_state		= indexedcopy(db.next_state,batchIndices,requestedTensorType) end
			if reqp.next_action then batch.next_action	= indexedcopy(db.next_action,batchIndices,requestedTensorType) end
			if reqp.reward then batch.reward 						= indexedcopy(db.reward,batchIndices,requestedTensorType) end
			if reqp.terminal then batch.terminal 				= indexedcopy(db.terminal,batchIndices,requestedTensorType) end
			batch.time_indices 													= indexedcopy(db.time_indices,batchIndices,requestedTensorType)
			batch.db_indices														= indexedcopy(db.db_indices,batchIndices,requestedTensorType)
	    return batch
		else
			print(db.time_indices[{{1,db.last_write_index}}])
			print(db.sequence_indices[{{1,db.last_write_index}}])
			print(startindices)
			print(endindices)

			print("No sequences of size " .. seq_properties.length .. " found.")

			return nil
		end

	else -- no sequence
		local availableIndices
		local batches = maxbatches
		if trainvaltest or surprises then
			local samplenumberstouse
			if trainvaltest then
				assert(self.splitindices, "The function setsplitfractions should be called to devide the database before a batch is requested of the train, validation or test parts.")
				if trainvaltest == 'train'	then --trainvaltest = nil, "train" , "validate" , "test"
					samplenumberstouse = self.splitindices.train:clone()
					batch.type = 'train data'
				elseif trainvaltest == 'validate' then
					samplenumberstouse = self.splitindices.validate:clone()
					batch.type = 'validation data'
				elseif trainvaltest == 'test' then
					samplenumberstouse = self.splitindices.test:clone()
					batch.type = 'test data'
				else
					print('trainvaltest was: ')
					print(trainvaltest)
					assert(false,'trainvaltest should be either nil, false, "train", "validate" or "test".')
				end
			else
				assert(self.surprising_memories,"Surprising memories not indicated yet")
				samplenumberstouse = self.experience_database.db_indices[self.surprising_memories]
			end
			if #samplenumberstouse:size()~=1 then
				print(samplenumberstouse)
				print("No samples to use, returning nil")
				return
			end
			if samplenumberstouse:nElement() < batch_size then
				samplenumberstouse = samplenumberstouse:repeatTensor(math.ceil(batch_size/samplenumberstouse:nElement()))
			end
			availableIndices 			= samplenumberstouse[samplenumberstouse:le(db.last_write_index)]
		else -- in most rl cases there is no need to split train and validation data as the dataset keeps changing.
			batch.type = 'any data'
			availableIndices 						= torch.linspace(1,db.last_write_index,db.last_write_index)
		end
		if prioritized and availableIndices:nElement() > batch_size*batches then -- rank based prioritized experience replay ( https://arxiv.org/abs/1511.05952 )
			--assert(batches==1,'prioritized experience replay returns only 1 batch since the priorities are updated for every TDE calculation')
			if not(self.per and self.per.indices and self.per.indices == availableIndices:nElement() and self.per.alpha == prioritized_alpha) then
				self.per = {}
				self.per.availableIndices = availableIndices
				local temp_indices = torch.linspace(1,availableIndices:nElement(),availableIndices:nElement())
				local sampling_probabilities = torch.cumsum(torch.ones(availableIndices:nElement()):cdiv(temp_indices):pow(prioritized_alpha))
					-- creating <batch_size> buckets of equal probability and sampling one experience uniformly from each
				self.per.indices = availableIndices:nElement()
				self.per.alpha = prioritized_alpha
				self.per.bucket_boundaries = torch.Tensor(1+batch_size*batches)
				self.per.is_weights = torch.Tensor(batch_size*batches)
				self.per.bucket_boundaries[1] = 0
				for bb = 1,batch_size*batches do
					local avi = availableIndices[sampling_probabilities:ge(sampling_probabilities[-1]*bb/(batch_size*batches))]
					local first_index
					if avi:nElement() > 0 then
						first_index = avi:min()
					else
						first_index = 0
					end
					self.per.bucket_boundaries[bb+1] = math.max(self.per.bucket_boundaries[bb]+1,
						first_index)
				end
			end

			local chosen_idcs = torch.Tensor(batch_size*batches)
			for bb = 1,batch_size*batches do
				chosen_idcs[bb] = math.random(self.per.bucket_boundaries[bb]+1,self.per.bucket_boundaries[bb+1])
				if countbasedis then
					self.per.is_weights[bb] = 1
					self.per.counts = self:get_extra_info('USECOUNT')
				else
					self.per.is_weights[bb] = (batch_size*batches)/self.per.indices * (self.per.bucket_boundaries[bb+1] - self.per.bucket_boundaries[bb])
				end
			end
			local tdevals = self:get_extra_info('TDE')
			tdevals = tdevals:index(1,self.per.availableIndices:long())
			local sorted_tdevals
			local sorted_indices
			sorted_tdevals, sorted_indices = torch.sort(tdevals,true)
			local availableIndices = self.per.availableIndices:index(1,sorted_indices:index(1,chosen_idcs:long()):long()):long()

			-- make sure each batch has spread out samples (devision in batches is done in indexedcopy)
			batchIndices = torch.LongTensor(batch_size*batches)
			local is_weight_indices = torch.LongTensor(batch_size*batches)
			temp_indices = torch.linspace(1,batch_size*batches,batch_size*batches)
			for i = 1,batchIndices:nElement() do
				batchIndices[i] = availableIndices:view(batch_size,batches)[1+((i-1)%batch_size)][math.ceil(i/batch_size)]
				if countbasedis then
					self.per.is_weights[i] = self.bintab[self.per.counts[batchIndices[i]]+1] or self.bintab[#self.bintab]
					--print(self.per.is_weights[i])
				end
				is_weight_indices[i] = temp_indices:view(batch_size,batches)[1+((i-1)%batch_size)][math.ceil(i/batch_size)]
			end

			if prioritized_beta > 0 then
				self.per.is_weights:pow(prioritized_beta)
				self.per.is_weights:div(self.per.is_weights:max())
				batch.is_weights =  indexedcopy(self.per.is_weights,is_weight_indices,requestedTensorType)
			end

		else
			local batch_size 						= math.min( batch_size,  availableIndices:nElement() )
			local ShuffledIndices 			= availableIndices:index(1,torch.randperm( availableIndices:nElement()):long())
			local sequence_length 			= 1
			if (one_batch) then -- TODO remove
				batchIndices 					= ShuffledIndices[{{1 ,  batch_size }}]:long()
			else
				batchIndices 					= ShuffledIndices[{{1 ,  batch_size*math.min(math.floor(availableIndices:nElement()/batch_size),batches) }}]:long()
			end
			-- should be merged with above! Situation of no PER but FIS
			if countbasedis then
				local allcounts = self:get_extra_info('USECOUNT')
				self.is_weights = batchIndices:clone():double()
				self.is_weight_indices = batchIndices:clone()
				for i = 1,self.is_weights:nElement() do
					self.is_weight_indices[i] = i
					self.is_weights[i] =  self.bintab[(allcounts[batchIndices[i]]+1)] or self.bintab[#self.bintab]
				end
			end
		end

		if reqp.observation then batch.observation 	= indexedcopy(db.observation,batchIndices,requestedTensorType) end
		if reqp.state then batch.state 							= indexedcopy(db.state,batchIndices,requestedTensorType) end
		if reqp.action then	batch.action						= indexedcopy(db.action,batchIndices,requestedTensorType) end
		if reqp.next_state then batch.next_state		= indexedcopy(db.next_state,batchIndices,requestedTensorType) end
		if reqp.next_action then batch.next_action	= indexedcopy(db.next_action,batchIndices,requestedTensorType) end
		if reqp.reward then batch.reward 						= indexedcopy(db.reward,batchIndices,requestedTensorType) end
		if reqp.terminal then batch.terminal 				= indexedcopy(db.terminal,batchIndices,requestedTensorType) end
		batch.time_indices 													= indexedcopy(db.time_indices,batchIndices,requestedTensorType)
		batch.db_indices														= indexedcopy(db.db_indices,batchIndices,requestedTensorType)
		if countbasedis and not(batch.is_weights) then

			self.is_weights:pow(prioritized_beta)
			self.is_weights:div(self.is_weights:max())

			batch.is_weights =  indexedcopy(self.is_weights,self.is_weight_indices,requestedTensorType)

		end
    return batch
  end
end

--devide the database in the given fractions
-- note that after calling this function minibatches can still be sampled without specifying train/test.
function em:setsplitfractions( trainfrac, validationfrac, testfrac )
	assert(trainfrac)
	assert(validationfrac)
	assert(testfrac)
	local sum = trainfrac+validationfrac+testfrac
	trainfrac = trainfrac / sum
	validationfrac = validationfrac / sum
	testfrac = testfrac / sum
	local allIndices = torch.randperm(self.experience_replay_size)
	self.splitindices = {}
	self.splitindices.train = torch.sort(allIndices[{{1,math.floor(trainfrac*self.experience_replay_size)}}])
	self.splitindices.validate = torch.sort(allIndices[{{math.floor(trainfrac*self.experience_replay_size)+1,math.max(math.floor(trainfrac*self.experience_replay_size)+1,math.floor((trainfrac+validationfrac)*self.experience_replay_size))}}])
	self.splitindices.test = torch.sort(allIndices[{{math.min(self.experience_replay_size,math.floor((trainfrac+validationfrac)*self.experience_replay_size)+1),self.experience_replay_size}}])
end

function em:update_extra_info(name,values,indices,special)
	assert(name and type(name=="string"),"No name provided")
	assert(values and indices,"Provide the new values and their database idices")
	if torch.isTensor(indices) then
		values = values:double()
		indices = indices:double()
	end
	assert(self.experience_database.extra_info,"extra info state table not initialized when the database was created")
	local info_state = false
	for i,state in ipairs(self.experience_database.extra_info) do
		if (state.name == name) then
			info_state = state
		end
	end
	assert(info_state, "info_state: " .. name .. " not found in the extra_info database")
	if torch.isTensor(values) then
		assert(values:nElement()==indices:nElement(),"Sizes of the values and indices do not match")
		local indx = indices:double():view(-1)
		local vals = values:double():view(-1)
		for i=1,indx:nElement() do
			if special and special == 'increment' then
				info_state.values[indx[i]] = info_state.values[indx[i]] + vals[i]
			else
				info_state.values[indx[i]] = vals[i]
			end
		end
	else
		if special and special == 'increment' then
			info_state.values[indices] = info_state.values[indices] + values
		else
			info_state.values[indices] = values
		end
	end
end



function em:get_extra_info(name)
 	assert(name and type(name=="string"),"No name provided")
	assert(self.experience_database.extra_info,"extra info state table not initialized when the database was created")
	local info_state = false
	for i,state in ipairs(self.experience_database.extra_info) do
		if (state.name == name) then
			info_state = state
		end
	end
	assert(info_state, "info_state: " .. name .. " not found in the extra_info database")
  return info_state.values:clone()
end

function em:update_surprise_flags(stats_to_use, verbal)
	local cats = {}
	local catnames = {}
	local existing_experiences = torch.ByteTensor(self.experience_replay_size):zero()
	existing_experiences[{{1,self.experience_database.last_write_index}}]:fill(1)
	if stats_to_use.DYNERROR and stats_to_use.RELDYNERROR then
		local dyne 					= self:get_extra_info('DYNERROR')
		local dynerv 				= dyne[dyne:lt(math.huge)]
		local dynemean 			= dynerv:mean()
		local reldyne 			= self:get_extra_info('RELDYNERROR')
		local highrelerror	= reldyne:gt(1):cmul(reldyne:lt(math.huge))
		highrelerror = highrelerror:cmul(existing_experiences)
		local baddyn = highrelerror:cmul(dyne:gt(dynemean))
		table.insert(cats, baddyn)
		table.insert(catnames, 'Bad dynamics predictions')
		if (not cats[0]) then
			cats[0] 		= dyne:ge(math.huge):cmul(existing_experiences)
			catnames[0] = 'Not used to train models yet'
		end
	end
	if stats_to_use.REWERROR then
		local rewe 				= self:get_extra_info('REWERROR')
		local rewerv 			= rewe[rewe:lt(math.huge)]
		local rewemean 		= rewerv:mean()
		local rewestd	 		= rewerv:std()

		table.insert(cats, (rewe:gt(rewemean+rewestd)):cmul(existing_experiences))
		table.insert(catnames, 'bad reward predictions')
		if (not cats[0]) then
			cats[0] 		= rewe:ge(math.huge):cmul(existing_experiences)
			catnames[0] = 'Not used to train models yet'
		end
	end

	if stats_to_use.STATEUNLIKELINESS then
		-- !!!! PARAMETER  TODO
		local UNLIKELINESS_THRESHOLD = 0.75
		local STDVWARN = 1e-2
		local unlike			= self:get_extra_info('STATEUNLIKELINESS')
		local unlikerv 		= unlike[unlike:lt(math.huge)]
		if unlikerv:std() < STDVWARN then
			print("--------------------- WARNING -----------------------")
			print("STATE UNLIKELINESS VALUES VERY SIMILAR, GAN COLLAPSE?")
			print("-----------------------------------------------------")
		end
		table.insert(cats, (unlike:gt(UNLIKELINESS_THRESHOLD):cmul(existing_experiences)))
		table.insert(catnames, 'States unlikely to be fantasized about')
		if (not cats[0]) then
			cats[0] 		= unlike:ge(math.huge):cmul(existing_experiences)
			catnames[0] = 'Not used to train models yet'
		end
	end

	local sm = cats[1]:clone()
	for i=2,#cats do
		sm = sm:add(cats[i])
	end
	sm = (sm:cmul(existing_experiences)):gt(0.5)

	self.surprising_memories = sm
	local total = existing_experiences:sum()
	if verbal then
		print("")
		print("Experience and fanatsy diagnostics: ")
		for i=0,#cats do
			print(catnames[i] .. ":  [ " .. cats[i]:sum() .. " / " .. total .. " ] ")
		end
		local perc = string.format("( %2.0f", 100*sm:sum()/total)
		print("TOTAL: [" .. sm:sum()  .. " / " .. total .. " ] " .. perc .. "%)")
	end
	return {problemfraction = sm:sum()/total}
end

function em:saveToFile(filename)
  torch.save(filename,self)
end

function em:save_mat_screenshot(filename,base_components,extra_info)
	save = {}
	if base_components then
		if base_components.time_indices then
			save.time_indices = self.experience_database.time_indices[{{1,self.experience_database.last_write_index}}]
		end
		if base_components.sequence_indices then
			save.sequence_indices = self.experience_database.sequence_indices[{{1,self.experience_database.last_write_index}}]
		end
		if base_components.state then
			save.state = self.experience_database.state[1][{{1,self.experience_database.last_write_index}}]
		end
		if base_components.next_state then
			save.next_state = self.experience_database.next_state[1][{{1,self.experience_database.last_write_index}}]
		end
		if base_components.action then
			save.action = self.experience_database.action[1][{{1,self.experience_database.last_write_index}}]
		end
		if base_components.reward then
			save.reward = self.experience_database.reward[{{1,self.experience_database.last_write_index}}]
		end
		if base_components.terminal then
			save.terminal = self.experience_database.terminal[{{1,self.experience_database.last_write_index}}]
		end
	end
	if(extra_info) then
		for i,name in pairs(extra_info) do
			local res = self:get_extra_info(name)
			save[name] = res[{{1,self.experience_database.last_write_index}}]
		end
	end
	mattorch.save(filename, save)
end


-- component = state, action, reward
-- property  = min, max, mean, std
function em:getComponentProperty(component, property)
	local comp
	if component == 'state' then
		comp = self.experience_database.state[1][{{1, self.experience_database.last_write_index}}]
	elseif component == 'action' then
		comp = self.experience_database.action[1][{{1, self.experience_database.last_write_index}}]
	elseif component == 'reward' then
		comp = self.experience_database.reward[{{1, self.experience_database.last_write_index}}]
	end
	if property == 'min' then
		return( comp:min(1) )
	elseif property == 'max' then
		return( comp:max(1) )
	elseif property == 'mean' then
		return comp:mean(1)
	elseif property == 'std' then
		return comp:std(1)
	end
end


local gs = torch.class('generalized_state')

function gs:__init(state_properties)
--name,experience_memory,dimension,statetype,prev_per_fullstate
	assert(state_properties.name and state_properties.experience_memory and state_properties.statetype,"Incomplete state_properties table" )
	self.state_properties = state_properties
	-- when a previous (or future) value of the state is requested that is not in the database the extrapolation strategy will be used to calculate the value based on the closest values that are in the database.
	assert(state_properties.extrap,"The extrapolation property state_properties.extrap is not set. Use 'nil','zero','zoh' or 'foh'")
	self.state_name 			          = state_properties.name
	self.expm 					            = state_properties.experience_memory
	assert(torch.type(state_properties.dimension)=="torch.LongStorage","Please use a torch.LongStorage to specify the state dimensions")
	self.state_dimension		        = state_properties.dimension or torch.LongStorage({1})
	self.state_memory_dimension     = torch.LongStorage(#self.state_dimension + 1)
	self.state_memory_dimension[1]  = state_properties.memory_size or self.expm.short_term_memory_size
	self.lastTimeIndex							= -1
	for i = 2,#self.state_memory_dimension do
		 self.state_memory_dimension[i] = self.state_dimension[i-1]
	end
	self.TimeIndices 			    = torch.Tensor(self.state_memory_dimension[1]):fill(-1)
	self.SequenceIndices 			=	torch.Tensor(self.state_memory_dimension[1]):fill(-1)
	self.currentWriteIndex 		= 1
	self.statetype 				    = state_properties.statetype
	self.prevPerFullState 		= state_properties.prev_per_fullstate or 0
	self.getFunction 			    = state_properties.getFunction
	self.extrapolation 			  = state_properties.extrap

	if self.statetype == "torch.DoubleTensor" then
		self.state_memory   = torch.Tensor(self.state_memory_dimension)
	elseif self.statetype == "torch.CudaTensor" then
		self.state_memory   = torch.CudaTensor(self.state_memory_dimension)
	elseif self.statetype == "number" then
		self.state_memory   = torch.Tensor(self.state_memory_dimension)
	else
		self.state_memory = {}
		for i = 1,self.state_memory_dimension[1] do
			self.state_memory[i] = {}
		end
	end
end

-- returns a copy of the part of the state that corresponds with the sequence_index, ordered by time_index.
function gs:ordered_seq(sequence_index)
	local nrIndices  = (#self.TimeIndices)[1]
	local memIndices = torch.linspace(1,nrIndices,nrIndices)
	local selectionMask = torch.eq(self.SequenceIndices,sequence_index)
	if selectionMask:sum() == 0 then
		print( self.state_name .. " : no information found for sequence " .. sequence_index )
		return nil
	end

	local relevantMemIndices = memIndices:maskedSelect(selectionMask)
	local relevantTimeIndices = self.TimeIndices:maskedSelect(selectionMask)
	local ti,ai = torch.sort(relevantTimeIndices)
	relevantMemIndices = relevantMemIndices:index(1,ai)
	return self.state_memory:index(1,relevantMemIndices:type('torch.LongTensor'))
end



function gs:receive_for_time(time_index,sequence_index,extra)
	assert(time_index,sequence_index,"provide the time and sequence indices")
	if (time_index <= self.lastTimeIndex) then
		return
	end
	self.lastTimeIndex = time_index
	assert(self.getFunction,"No get function set.")
	local newstate = self:getFunction(time_index,extra)
	if newstate then
		if self.TimeIndices[self.currentWriteIndex] < time_index then
			self.TimeIndices[self.currentWriteIndex] = time_index
			self.SequenceIndices[self.currentWriteIndex] = sequence_index
			if type(newstate)=="number" then
				self.state_memory[self.currentWriteIndex] = newstate
			else
				self.state_memory[self.currentWriteIndex]:copy(newstate)
			end

			self.currentWriteIndex = self.currentWriteIndex + 1
      if self.currentWriteIndex > self.TimeIndices:size(1) then
        self.currentWriteIndex = 1
      end
    end
  else
    if (self.TimeIndices[self.currentWriteIndex] ~= -1) then
        self.TimeIndices[self.currentWriteIndex] = -1
    end
  end
end

-- obtain a value from the short term memory, assumes the value is already in the memory. Use receive_for_time to obtain the value for the current time index.
function gs:get_value_for_timestep(time_index,sequence_index,extra)

	local returnvalue = function(state,index)
			if torch.isTensor(state.state_memory[index]) then
				return state.state_memory[index]:clone()
			else
				return state.state_memory[index]
			end
	end

	if (time_index > self.lastTimeIndex) then
		self:receive_for_time(time_index,sequence_index,extra)
	end



  -- shortcut for the most common case of having just received the required value.
  if (self.currentWriteIndex-1 > 0 and self.TimeIndices[self.currentWriteIndex-1] == time_index) then
    return  returnvalue(self,self.currentWriteIndex-1)
  end
-- start looking from current value, go back to start, resume from end untill current value
-- if not found use the extrapolation policy.
	--self.extrapolation
  local arrayIndex      = self.currentWriteIndex
  local closestindex    = self.currentWriteIndex
  local closesTimeDelta = math.huge
  local found = false
  while true do
    if (self.TimeIndices[arrayIndex] == time_index) then
      found = true
      break
    else
      if (math.abs(self.TimeIndices[arrayIndex] - time_index) < closesTimeDelta) then
        closestindex    = arrayIndex
        closesTimeDelta = math.abs(self.TimeIndices[arrayIndex] - time_index)
      end
      arrayIndex = arrayIndex - 1
      if (arrayIndex < 1) then
        arrayIndex = self.TimeIndices:size(1)
      end
      if arrayIndex == self.currentWriteIndex then -- had whole loop
        break
      end
    end
  end
  if found then
    return returnvalue(self,arrayIndex)
  else
    -- extrapolate
    if self.extrapolation == 'nil' then
      return nil
    elseif self.extrapolation == 'zero' then
      return self.state_memory[closestindex]:clone():zero()
    elseif self.extrapolation == 'zoh' then
      return returnvalue(self,closestindex)
    elseif self.extrapolation == 'foh' then
      assert(false,"FOH not yet implemented")
      --TODO: implement FOH
    end
  end
assert(false,"time_index not found and no correct extrapolation policy set.")
end

--function gs:__index(time_index)
--	return 0 --self:get_value_for_timestep(time_index)
--end
