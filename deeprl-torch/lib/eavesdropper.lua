--[[
		Tim de Bruin 2015
		deepRL-torch
		communicator class.
		Takes care of communication between switchboards and the environment.
--]]

--[[ communication protocol: 
	
	seperator 			= '|' (also end with |!)
	first part 			= transmission name
	second part 		= transmission index (int)
	remaining parts 	= 1 letter which indicates the type of the message part:
							n = new part name, indicates part name for all remaining parts untill the next part name
							s = string 
							f = float
							i = integer
							o = serialized object
						, rest of message untill following |
--]]

-- TODO: IPC, remote server

--require('mobdebug').start()
require "torch"
require "zhelpers"
local zmq = require "lzmq"
require 'sys'

ADDRESS = "tcp://localhost:5555"

context 			        = zmq.context() 
subscriber, err 		= context:socket{zmq.SUB,
		  subscribe = "";
		  connect   = ADDRESS;
		}
    zassert(subscriber, err)

print("Listening in on " .. ADDRESS)
print("--------------------------------")
print("")
while true do
	message = subscriber:recv()
	print(message)
end

