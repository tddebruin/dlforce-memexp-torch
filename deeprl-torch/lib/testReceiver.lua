require "torch"
require "zhelpers"
local zmq = require "lzmq"
require 'sys'

-- load lua-pb first.
require"pb"

-- now you can use require to load something.proto
require"drl_messages"


ADDRESS = "tcp://localhost:5555"
  
  
context = zmq.context()
assert(context)

-- Socket to receive signals
subscriber, err 		= context:socket{zmq.SUB,
	  subscribe = "";
	  connect   = ADDRESS;
    conflate  = 1;
	}
	
publisher, err 		= context:socket{zmq.PUB,
	  bind   = "tcp://*:5562"
	}

			
zassert(subscriber, err)
zassert(publisher, err)

function sleep(s)
  local ntime = os.time() + s
  repeat until os.time() > ntime
end

local isSync = false
local stateDimension, actionDimension
local state_pos 
local state_vel
while isSync == false do
  local syncMess = DRL_MESSAGES.drl_unimessage{type = DRL_MESSAGES.drl_unimessage.Type.MESSTR, time_index = 1,msgstr="senddim"}
  publisher:send(syncMess:Serialize())

  local msg = subscriber:recv(zmq.DONTWAIT)

  if msg then
    local msg_obj = DRL_MESSAGES.drl_unimessage():Parse(msg)
    if msg_obj.type == "DIMENSION" then
      stateDimension  = msg_obj.dimension.state[1]
      actionDimension = msg_obj.dimension.action[1]
      isSync = true
    end
    print("message received" .. msg_obj.type)
  end
  
end

print("state dimension:" .. stateDimension .. " action dimension:" .. actionDimension)

while true do

  local messageSend = DRL_MESSAGES.drl_unimessage()
  local action = torch.randn(2)
  
  local message_action =  DRL_MESSAGES.drl_unimessage.Action()
  message_action.actions = action:totable()
  messageSend.action = message_action
  messageSend.type=DRL_MESSAGES.drl_unimessage.Type.CONTROLACTION
  messageSend.time_index = 1
  publisher:send(messageSend:Serialize())
    
  sleep(1)

  local msg = subscriber:recv(zmq.DONTWAIT)

  if msg then
    local msg_obj = DRL_MESSAGES.drl_unimessage():Parse(msg)
    if msg_obj.type == "ROBOTSTATE" then
      state_pos = torch.Tensor(msg_obj.robotstate.state_pos)
      state_vel = torch.Tensor(msg_obj.robotstate.state_velocity)
      print(state_pos)
      print(state_vel)
    end
    print("message received" .. msg_obj.type)
  end
end

  
 
