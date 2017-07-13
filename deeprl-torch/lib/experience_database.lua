--[[
		Tim de Bruin 2015
		deepRL-torch
		experience database class.
		Constructs the full RL state from its parts and / or 
		maintains the experience database for training,
		
--]]


local experience_database = torch.class('experience_database')

function experience_database:__init(stateReconstruction, database, settings)
	self.settings = settings
	if stateReconstruction then
		self.reconstructionDB = {
			xpIndex = torch.Tensor(self.settings.DB.state_reconstruction.buffersize),
			stateComponents = {}
		}
		for sc = 1,#self.settings.RL.state do
			table.insert(self.reconstructionDB.stateComponents,torch.Tensor(self.settings.DB.state_reconstruction.buffersize,self.settings.RL.state[sc].dimension))
		end	
	end
	
	if database then
	
	end
end
	
function experience_database:loadFromFile()
	
end


