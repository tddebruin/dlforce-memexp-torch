--[[
		Tim de Bruin 2015
		deepRL-torch
		antenna class.
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


require "torch"
require "zhelpers"
local zmq = require "lzmq"
require 'sys'

local antenna = torch.class('antenna')
 
function antenna:__init(antenna_settings)
	self.sendOrReceive 		= antenna_settings.sendOrReceive
	self.transmissionname 	= antenna_settings.transmissionname
	self.integerformat 		= antenna_settings.floatformat or "%+1.7e"
	self.floatformat 		= antenna_settings.integerformat or "%+i"
	self.lastIndex 			= 0
	self.messageStore 		= {}	
	self.context 			= zmq.context() 
	if (self.sendOrReceive == 'send') then
		self.publisher, err = self.context:socket{zmq.PUB, bind = antenna_settings.bind or "tcp://*:5555"}
		zassert(self.publisher, err)
	elseif (self.sendOrReceive == 'receive') then
		self.subscriber, err = self.context:socket{zmq.SUB,
		  subscribe = self.transmissionname .. " ";
		  connect   = antenna_settings.connect or "tcp://localhost:5555";
		}
		zassert(self.subscriber, err)	
	else
		print('Not sending or receiving...')
	end
end

local receive = function(antenna)
	more = true
	while more do
		local messageString, more = antenna.subscriber:recv()
		print(messageString)
		antenna:interpret(messageString)	
	end	
end

local interpret = function(messageString)
	counter = 1
	local currentName 	= false
	local currentObject = false
	local addedToPrevMessage = false
	
	local messageParts = {}
	for w in string.gmatch(messageString, "(%w+)|") do
		if counter == 1 then
			assert(w==self.transmissionname)
		elseif counter == 2 then
			messageParts.time_index = tonumber(i)
			for i,m in ipairs(self.messageStore) do
				if m.time_index == messageParts.time_index then
					messageParts = m.messageParts
					addedToPrevMessage = true
					break
				end
			end
		else
			local firstLetter = i:sub(1,1)
			local remainder = i:sub(2,#i)
			assert(firstLetter and remainder,"bad string: " .. i .. " , " .. counter .. " part of: " .. messageString )
			if firstLetter == "n" then -- name
				currentName = remainder
			elseif firstLetter == "s" then -- string
				currentObject = remainder
			elseif firstLetter == "f" then -- float
				currentObject = tonumber(remainder)
			elseif firstLetter == "i" then -- integer
				currentObject = tonumber(remainder)
			elseif firstLetter == "o" then -- serialized lua/torch object
				currentObject = torch.deserialize(remainder)
			else
				currentObject = false
			end
			if (currentObject) then
				assert(type(currentName)=="string")
				local found = false
				for ci = 1,#messageParts do
					if (messageParts[ci].name == currentName) then
						found = true
						table.add(messageParts[ci].content,currentObject)
						break
					end
				end
				if not(found) then					
					table.add(messageParts,{name = currentName, time_index = messageParts.time_index, content={currentObject}})
				end
			end
		end
		counter = counter + 1				
	end
	if not(addedToPrevMessage) then
		table.insert(self.messageStore,messageParts)
	end
end

local extract_message_part = function(object_name)
	for m,message in ipairs(self.messageStore) do
		for i,messagePart in ipairs(message) do
			if messagePart.name == object_name then
				local response = table.remove(message,i)
				if #message == 0 then
					table.remove(self.messageStore,m)
				end
				return response
			end
		end
	end	
	return nil
end


function antenna:get(local_time_index,object_name)
	if local_time_index > self.lastIndex then
		receive(self)
		self.lastIndex = local_time_index
	end
	local wrappedMessage = self:extract_message_part(object_name)
	if wrappedMessage then
		if #wrappedMessage.content == 1 then
			return wrappedMessage.content[1], wrappedMessage.time_index
		else
			-- check if all message contents are numbers, make tensor if true
			local allNumbers = true
			for i=1,#wrappedMessage.content do
				if not( type(wrappedMessage.content[i])=="number" ) then
					allNumbers = false
				end
			end	
			if allNumbers then
				return torch.Tensor(wrappedMessage.content), wrappedMessage.time_index
			else
				return wrappedMessage.content, wrappedMessage.time_index
			end
		end		
	end
	return nil
end

local transmit = function(antenna,transmissionstring)
	print(transmissionstring)
	antenna.publisher:send(transmissionstring)	
end


local tableToString = function(antenna,table)
	local transmission_string = antenna.transmissionname .. " |" .. table.time_index .. "|" 
	for i,messagepart in ipairs(table.messageParts) do
		transmission_string = transmission_string .. messagepart.name .. "|" 
		-- encode the string based on the transmissiontype
		if messagepart.transmissionType == "s" then
			for mp = 1,#messagepart do 
				transmission_string = transmission_string .. messagepart[i] .. "|"
			end
		elseif (messagepart.transmissionType == "f" or messagepart.transmissionType == "i") then
			if messagepart.transmissionType == "i" then
				antenna.numberformat = antenna.integerformat	
			else
				antenna.numberformat = antenna.floatformat
			end
			for mp = 1,#messagepart do 
				if torch.type(messagepart[mp])=="number" then
					transmission_string = transmission_string .. messagepart.transmissionType .. string.format(antenna.numberformat, messagepart[mp]) .. "|"
				elseif torch.type(messagepart[mp])=="table" then
					for a in pairs(messagepart[mp]) do
						transmission_string = transmission_string .. messagepart.transmissionType .. string.format(antenna.numberformat, a) .. "|"
					end
				else 
					if (torch.type(messagepart[mp])):find("Tensor") then
						for a = 1,messagepart[mp]:nElement() do
							transmission_string = transmission_string .. messagepart.transmissionType .. string.format(antenna.numberformat, messagepart[mp][a]) .. "|"
						end
					end
				end
			end			
		elseif messagepart.transmissionType == "o" then
			transmission_string = transmission_string .. messagepart.transmissionType .. torch.serialize(messagepart[i]) .. "|"		
		else
			assert(false,"unknown transmissiontype: " .. messagepart.transmissionType)
		end
	end 
return transmission_string
end



--- transmissionType is one of the s,f,i,o types discussed in the communication protocol section
function antenna:send_one(time_index,object,object_name,transmissionType)
	assert(time_index and object and object_name and transmissionType)
	local transmission_message_table = {time_index = time_index, messageParts={{name = object_name, transmissionType = transmissionType, object}}}
	transmit(self,tableToString(self,transmission_message_table))	
end

--- Send multiple objects in one message. 
function antenna:send_multiple(time_index,object_table)
	assert(false,"needs to be implemented")	
	
	
end








