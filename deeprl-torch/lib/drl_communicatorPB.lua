--[[
		Tim de Bruin 2015
		deepRL-torch
		communicator class.
		Takes care of communication between switchboards and the environment.
--]]

--- Communicator class
-- This class 



-- TODO: IPC, remote server

--require('mobdebug').start()
require "torch"
require "zhelpers"
local zmq = require "lzmq"
require 'sys'
require 'pb' 
require 'drl_messages'

local interpret = function(channel,messageString)
	local msg_obj = DRL_MESSAGES.drl_unimessage():Parse(messageString)
	if (not channel.communicator.mastercomm and msg_obj.time_index > channel.communicator.time_index) then
		channel.communicator.time_index = msg_obj.time_index
	end
	if msg_obj.type == "DIMENSION" then
		channel.dimension = msg_obj.dimension
		print("Dimension:")
		for i,dimcomponent in ipairs(channel.dimension.component) do
			if (dimcomponent.component_name == 'state') then
				channel.statedimension = torch.LongStorage(dimcomponent.component_dimension)
				print( "State: ")
				print(channel.statedimension)
				channel.state_bounds = {
					min = dimcomponent.bound_min,
					max = dimcomponent.bound_max,
				}
			elseif dimcomponent.component_name == 'action' then 
				channel.actiondimension = torch.LongStorage(dimcomponent.component_dimension)
				channel.action_type = dimcomponent.component_type
				channel.action_bounds = {
					min = dimcomponent.bound_min,
					max = dimcomponent.bound_max,
				}
				print( "Action: ")
				print(channel.actiondimension)
			elseif dimcomponent.component_name == 'observation' then
				channel.observationdimension = torch.LongStorage(dimcomponent.component_dimension)		
				print( "Observation: ")
				print(channel.observationdimension)
			else
				print('Warning: unknown dimension component name: ' .. dimcomponent.component_name)
			end
		end
		
	elseif msg_obj.type == "MESSTR" then
		-- process message
	elseif msg_obj.type == "STATEPART" then
		assert(msg_obj.name,"Unnamed state part")
		channel.communicator.messageStore.messageParts[msg_obj.name] = msg_obj
	elseif msg_obj.type == "REWARDTERMINAL" then		
		channel.communicator.messageStore.terminalreward = msg_obj.rwt
	elseif msg_obj.type == "RLSTATE" then
			-- TODO:
	else
			assert(false, "Unrecognized message type")
	end
		return msg_obj
end			
		

local receive = function(channel)
	while true do
    messageString = channel.subscriber:recv(zmq.DONTWAIT)
		if (messageString) then
      	interpret(channel,messageString)
    else
      break
    end
	end	
end


local com = torch.class('communicatorPB')
 
function com:__init(communicator_settings,mastercomm)
  self.mastercomm           		= mastercomm -- boolean, true for controller mod
  self.communicator_settings    = communicator_settings or {}  
	self.time_index 							= -1			    
	self.sampling_time        		= self.communicator_settings.sampling_time or 0
	self.messageStore 						= {}
	self.messageStore.messageParts= {}
	self.messageStore.dimensions 	= {}
	self.messageStore.RLstates 		= {}
	self.messageStore.terminalreward = {}	
	self.context 			    				= zmq.context() 
  self.channels         				= {}
  self.lastPauseTime				    = 0
  self.lastLoad	 								= 0
end

function com:close()
	self.context:term()
end

function com:get_context()
	return self.context
end

function com:get_time_index() 
	return self.time_index
end

function com:advance_sequence_index()
	if self.sequence_index then
		self.sequence_index = self.sequence_index + 1
	else
		self.sequence_index = 1
	end
end

function com:get_sequence_index()
	if self.sequence_index then
		return self.sequence_index 
	else
		return -1
	end
end



function com:wait_for_synch()
	print("Waiting for synchronization on the following channels:")
	for idx, channel in ipairs(self.channels) do
		if not(channel.synched) then
			print(channel.channel_settings.name)
		end
	end
	print("")	
	while true do
		local allSynched = true
		if self.mastercomm then
			for idx, channel in ipairs(self.channels) do
				if not(channel.synched) then
					allSynched = false

					channel:send_synch_req()
					messageString = channel.subscriber:recv(zmq.DONTWAIT)
					while messageString do
						if (DRL_MESSAGES.drl_unimessage():Parse(messageString)).type == "DIMENSION" then
							interpret(channel,messageString)
							if not channel.synched then
								channel.synched = true
								print("Channel synchronized:  " .. channel.channel_settings.name)
								channel:send_synch_success()
							end
						end
						messageString = channel.subscriber:recv(zmq.DONTWAIT)	
					end
				end
			end
		else -- not master, wait for signal then respond
			--TODO: implement
			assert(false,"Not yet implemented")
		end
		if allSynched then
			break
		else
      self:sleep(0.01)
		end	
	end
end

function com:cleanstart()
	self:sleep(0.1)
	for idx, channel in ipairs(self.channels) do
		for i=1,3 do-- first message will be missed, second message to close, third one is for good luck.
			channel:send_exit()
		end
	end
	
	if self.envcommand then 
		print('executing: ' .. self.envcommand)
		os.execute(self.envcommand)
		self:sleep(1)
		self:wait_for_synch()
	end
	communicator:advance_sequence_index()
end

function com:sleep(seconds)
  local t = torch.Timer()
  while t:time().real < seconds do
    --os.sleep(0.00001)
  end	
end

function com:advance_clock()
  if self.mastercomm then
    self.time_index = self.time_index + 1
    for idx, channel in ipairs(self.channels) do
      channel:send_time()
    end
  end
  
  --collectgarbage()  
  if self.clock then  
    local idle_period  = self.sampling_time - self.clock:time().real 
    self.lastPauseTime = idle_period
    self.lastLoad	 		 = (self.sampling_time - idle_period)/self.sampling_time	
    if idle_period > 0 then
      self:sleep(idle_period)
    end   
    self.clock:reset()
  else
    self.clock = torch.Timer()
  end
end 

--- sets the time_index to one and broadcasts it, if this is the master clock.
function com:reset_clock()
  if self.mastercomm then	
		self.time_index = 1
		for idx, channel in ipairs(self.channels) do
		  channel:send_time()
		end	
	end
end

function com:getLastLoad()
	return self.lastLoad,self.lastPauseTime
end
function com:receive_all()
    for idx, channel in ipairs(self.channels) do
      receive(channel,true)
    end  
end

function com:add_channel(channel_settings)
  local newchannel
  if channel_settings.internal then
    newchannel = internal_channel(self,channel_settings)
  else
    newchannel = channel(self,channel_settings)
	end
  table.insert(self.channels,newchannel)
  return newchannel
end

local ch = torch.class('channel')

function ch:__init(communicator,channel_settings)
  self.synched 			  		= false
  self.communicator       = communicator
  self.channel_settings   = channel_settings
  assert(self.channel_settings.name,"no channel settings or name given")
  self.lastIndex          = -1
  if communicator.mastercomm then
    self.publisher, err = self.communicator.context:socket({zmq.PUB, bind = channel_settings.bindpub or "tcp://*:5555"})
		self.subscriber, err = self.communicator.context:socket{zmq.SUB,
		  subscribe = "";
		  bind = channel_settings.bindsub or "tcp://*:5556"
		}
    zassert(self.publisher, err)
    zassert(self.subscriber, err)	
    
  else
    self.publisher, err = self.communicator.context:socket{zmq.PUB, connect   = channel_settings.connectpub or "tcp://localhost:5556";}
		self.subscriber, err = self.communicator.context:socket{zmq.SUB,
		  subscribe = "";
		  connect   = channel_settings.connectsub or "tcp://localhost:5555";
		}
    zassert(self.publisher, err)
    zassert(self.subscriber, err)
  end
end

function ch:receive_n_fortimeindex(n,time_index)
	local counter = 0
	local t = torch.Timer()
	local TIMEOUT = 30
	while true do
    messageString = self.subscriber:recv(zmq.DONTWAIT)
		if (messageString) then
      	obj = interpret(self,messageString)
      	if obj.time_index == time_index then
      		counter = counter + 1
      	end
      	if counter == n then
      		return
      	end
	  else
	  	if t:time().real > TIMEOUT then
	  		print("ENVIRONMENT TIMEOUT!")
	  		print("RESETTING ENVIRONMENT")
	  		self.communicator:cleanstart()
	  		return 
	  	end
		end	
	end
end



function ch:send_synch_req()
	local syncMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.MESSTR, time_index = self.communicator.time_index, msgstr="senddim"}
		
		self.publisher:send(syncMess:Serialize())	
end

function ch:send_synch_success()
	local syncMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.MESSTR, time_index = self.communicator.time_index, msgstr="synched"}
		self.publisher:send(syncMess:Serialize())	
end

function ch:send_env_reset()
	local syncMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.MESSTR, time_index = self.communicator.time_index, msgstr="envreset"}
		self.publisher:send(syncMess:Serialize())	
end

function ch:send_env_render()
	local syncMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.MESSTR, time_index = self.communicator.time_index, msgstr="envrender"}
		self.publisher:send(syncMess:Serialize())	
end

function ch:send_exit()
	local syncMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.MESSTR, time_index = self.communicator.time_index, msgstr="exit"}
		self.publisher:send(syncMess:Serialize())	
end


function ch:send_time()
	local syncMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.MESSTR, time_index = self.communicator.time_index, msgstr="time_index"}
		self.publisher:send(syncMess:Serialize())	
end

function ch:send_state_part(xname, x, dx, ddx, dddx)
	local stateMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.STATEPART, time_index = self.communicator.time_index, name = xname, state = x, first_derivative = dx, second_derivative = ddx, third_derivative = dddx}
		self.publisher:send(stateMess:Serialize())	
end

function ch:send_action(aname, a)
	local message_action =  DRL_MESSAGES.drl_unimessage.Action()
	message_action.actions = a:totable() 
	local stateMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.CONTROLACTION, time_index = self.communicator.time_index, action = message_action}
	self.publisher:send(stateMess:Serialize())	
end


function ch:get_state_part(local_time_index,object_name,derivorder)
	local dorder = derivorder or "state"
	if local_time_index > self.lastIndex then
		receive(self)
		self.lastIndex = local_time_index
	end
	local msg = self.communicator.messageStore.messageParts[object_name] 
	if msg then
		local table = msg.statepart[dorder]
		if table then 
			return torch.Tensor(table)
		end
	end
	return nil
end

function ch:get_reward(local_time_index)
	if local_time_index > self.lastIndex then
		receive(self)
		self.lastIndex = local_time_index
	end
	local msg = self.communicator.messageStore.terminalreward
	if msg then
		return msg.reward
	end
	return nil
end

function ch:get_terminal(local_time_index)
	if local_time_index > self.lastIndex then
		receive(self)
		self.lastIndex = local_time_index
	end
	local msg = self.communicator.messageStore.terminalreward 
	if msg then
		return msg.terminal
	end
	return nil
end



-----------------------------------------------------------------------------------------
-- INTERNAL FAKE CHANNEL
-----------------------------------------------------------------------------------------
local chi = torch.class('internal_channel')

function chi:__init(communicator,channel_settings)
  self.synched 			  		= true
  self.communicator       = communicator
  self.channel_settings   = channel_settings
  self.lastIndex          = -1
  self.env 								= channel_settings.experiment
  local envprops 					= self.env:get_bounds()
  self.statedimension			= torch.LongStorage({envprops.state.dimension}) 
  self.state_bounds 			= {
  	min = envprops.state.min,
  	max = envprops.state.max,
	}
	self.actiondimension = torch.LongStorage({envprops.action.dimension})
	self.action_bounds = {
  	min = envprops.action.min,
  	max = envprops.action.max,	
  }
  self.action_type = envprops.action.action_type
end

function chi:receive_n_fortimeindex(n,time_index)
end

function chi:send_synch_req()
	assert(self.env)
end

function chi:send_synch_success()
end

function chi:send_env_reset(scenario)
	self.env:reset(scenario)
end

function chi:send_env_render()
end

function chi:send_exit()	
end


function chi:send_time()
end

function chi:send_state_part(xname, x, dx, ddx, dddx)
end

function chi:send_action(aname, a)
		self.env:step(a)
end

function chi:get_state_part(local_time_index,object_name,derivorder)
	return self.env:get_state()
end

function chi:get_reward(local_time_index)
	return self.env:get_reward()
end

function chi:get_terminal(local_time_index)
	return self.env:get_terminal()
end

function chi:reward_function(state, action, next_state)
	return self.env:reward_function(state, action, next_state)
end

function chi:uniform_random_state()
	return self.env:uniform_random_state()
end	
