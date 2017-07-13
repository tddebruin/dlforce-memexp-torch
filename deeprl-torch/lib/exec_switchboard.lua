--[[
		Tim de Bruin 2015
		deepRL-torch
		executor-switchboard class.
		Takes care of communication between other classes,
		Executes a policy, does not do training
--]]
require 'switchboard'
require 'experience_database'

local exec_switchboard, parent = torch.class('exec_switchboard','switchboard')

function exec_switchboard:__init(settings)
	parent.__init(self,settings)
end

function exec_switchboard:setup()	
	assert(self.settings,"settings table not set")
	self.edb 			= experience_database(true,false,self.settings)
end

function exec_switchboard:episode()
	

end


