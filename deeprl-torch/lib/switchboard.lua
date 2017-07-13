--[[
		Tim de Bruin 2015
		deepRL-torch
		switchboard class.
		Takes care of communication between other classes.
--]]



local switchboard = torch.class('switchboard')
 
function switchboard:__init(settings)
	self.settings = settings

end





