--[[
		Tim de Bruin 2015
		Deep Reinforcement Learning
		Logging, diagnostics and plotting functions
--]]

require "gnuplot"

local diag = torch.class('drl_diagnostics')

function diag:__init(settings)
	self.settings = settings
	self.monitored_states = {}
	self.console_summary_per_sequence = {}
	self.console_summary_per_sequence_settings = {}
	self.plots = {}
	self.itorch_plots = {} -- deprecated
	self.gnuplots = {} -- deprecated
  	self.intensity_maps = {} -- deprecated
	self.lastSequence = 1
end

function diag:addStatePlot(state_table,settings_table)
	-- TODO add to monitoring
	assert(state_table,"state_table not given, provide a table with the GeneralizedState s that need to be plotted")
	assert(settings_table,"settings_table not given, provide a table with the settings for the plot")
	
	gnuplot.figure(1)
	gnuplot.plot(state_table[1].state_memory)
	gnuplot.title(settings_table.name)
	gnuplot.xlabel('Time index')
	gnuplot.ylabel('Value')
	
end	



function diag:addConsoleSummary(state,settings_table)
	assert(state,"No state prvided")
	assert(settings_table,"No settings_table provided")
	
	-- TODO: every Nth (sequence/ time_index etx)
	table.insert(self.console_summary_per_sequence,state)
	table.insert(self.console_summary_per_sequence_settings,settings_table)
	table.insert(self.monitored_states,state)
	
end

function diag:addGNULinePlotPerTimestep(settings)
	print("diag:addGNUPlotPerEpisode is deprecated")
	assert(settings,"no plot details provided" )
	assert(settings.xdata, "Settings table needs a xdata state for the horizontal axis of the plot")
	assert(settings.ydata, "Settings table needs a ydata state for the vertical axis of the plot")
	if (#settings.xdata.state_memory:squeeze():size() > 1) then
		assert( settings.xdim, "x-state is more than 1 dimensional, provide dimensions to use for the plot in xdim") 
		assert( #settings.xdim == #settings.xdata.state_memory:size() -1,"Dimension of ydim is unequal to the dimension of the state: ydim: " ..  #settings.xdim .. ", state (excluding time: " .. #settings.xdata.state_memory:size() -1)
	end
	settings.figureid  = #self.gnuplots + 1
  	settings.dontskipme = true
	if settings.figureid == 1 then
		gnuplot.closeall()
	end
	table.insert(self.gnuplots,settings)
	table.insert(self.monitored_states,settings.xdata)
	table.insert(self.monitored_states,settings.ydata)

end

--TODO: remove
function diag:addGNUPlotPerEpisode(settings)
	print("diag:addGNUPlotPerEpisode is deprecated")
	settings.figureid  = #self.gnuplots + 1
	if settings.figureid == 1 then
		gnuplot.closeall()
	end
	settings.eptensor = true
	assert(settings.epdata,"A 1D epdata tensor is needed")
  	settings.dontskipme = true
	table.insert(self.gnuplots,settings)
end

--TODO: remove
function diag:addItorchPlotPerTimestep(settings)
	Plot = require 'itorch.Plot'
	assert(settings,"no plot details provided" )
	assert(settings.xdata, "Settings table needs a xdata state for the horizontal axis of the plot")
	assert(settings.ydata, "Settings table needs a ydata state for the vertical axis of the plot")
	if (#settings.xdata.state_memory:squeeze():size() > 1) then
		assert( settings.xdim, "x-state is more than 1 dimensional, provide dimensions to use for the plot in xdim") 
		assert( #settings.xdim == #settings.xdata.state_memory:size() -1,"Dimension of xdim is unequal to the dimension of the state: xdim: " ..  #settings.xdim .. ", state (excluding time: " .. #settings.xdata.state_memory:size() -1)
	end
	if (#settings.ydata.state_memory:squeeze():size() > 1) then
		assert( settings.ydim, "y-state is more than 1 dimensional, provide dimensions to use for the plot in ydim") 
		assert( #settings.ydim == #settings.ydata.state_memory:size() -1,"Dimension of ydim is unequal to the dimension of the state: ydim: " ..  #settings.ydim .. ", state (excluding time: " .. #settings.ydata.state_memory:size() -1)
	end
	settings.plot = Plot():legend(false):title(settings.xdata.state_name .. ' vs ' .. settings.ydata.state_name)
	settings.plotdrawn = false
	table.insert(self.itorch_plots,settings)
	table.insert(self.monitored_states,settings.xdata)
	table.insert(self.monitored_states,settings.ydata)
	
end

function diag:addItorchPlotPerEpisode(settings)
	settings.plot = Plot():legend(false):title(settings.name)
	settings.plotdrawn = false 
	settings.eptensor = true
	assert(settings.epdata,"A 1D epdata tensor is needed")
	table.insert(self.itorch_plots,settings)
end



local print_consolesum_seq = function(state,settings,sequence)
	assert(sequence)
	assert(state)
	assert(settings)
	local occcount 	= 0
	local sum 			= 0
	local min				= math.huge
	local max 			= -math.huge
	for i=1,state.SequenceIndices:nElement() do
		if (state.SequenceIndices[i]==-1) then
			
			break
		elseif (state.SequenceIndices[i]==sequence) then	 		
			occcount = occcount + 1
			value = state.state_memory[i]
			if (torch.isTensor(value)) then
				value = value:mean()
			end
			sum = sum + value
			if (value < min) then 
				min = value
			end
			if (value > max) then 
				max = value
			end
		end
	end
	if (occcount > 0) then
		print(string.format("%s [%d samples]: mean: %.5E, min: %.4E, max %.4E.", state.state_name, occcount, sum/occcount , min , max))
	else
		print(state.state_name .. " has no measurements for sequence " .. sequence)
	end
end

function diag:get_state_average_for_sequence(state,sequence)
	assert(sequence)
	assert(state)
	local occcount 	= 0
	local sum 			= 0
	for i=1,state.SequenceIndices:nElement() do
		if (state.SequenceIndices[i]==-1) then
			
			break
		elseif (state.SequenceIndices[i]==sequence) then	 		
			occcount = occcount + 1
			value = state.state_memory[i]
			if (torch.isTensor(value)) then
				value = value:mean()
			end
			sum = sum + value
		end
	end	
	return sum/occcount
end

function diag:get_state_sum_for_sequence(state,sequence)
	assert(sequence)
	assert(state)
	local occcount 	= 0
	local sum 			= 0
	for i=1,state.SequenceIndices:nElement() do
		if (state.SequenceIndices[i]==-1) then
			
			break
		elseif (state.SequenceIndices[i]==sequence) then	 		
			occcount = occcount + 1
			value = state.state_memory[i]
			if (torch.isTensor(value)) then
				value = value:mean()
			end
			sum = sum + value
		end
	end	
	return sum
end



function diag:update(time_index,sequence_index)
	if sequence_index < 1 then 
    return
  end
  for index, state in ipairs(self.monitored_states) do
		state:receive_for_time(time_index,sequence_index)
	end
	for index, map in ipairs(self.intensity_maps) do
    	map:update(time_index,sequence_index)
  	end
  	for index,plot in ipairs(self.plots) do
      plot:update(time_index,sequence_index)
  	end
	


	if sequence_index > self.lastSequence then
		-- console summary
		if (#self.console_summary_per_sequence > 0 and not(itorch)) then
			print(" ")
			print("Sequence " .. self.lastSequence .. " summary:")
			for i,j in ipairs(self.console_summary_per_sequence) do
				print_consolesum_seq(j,self.console_summary_per_sequence_settings[i],self.lastSequence)
			end
		end
	
		-- itorch plots
		for index, itorchplot in ipairs(self.itorch_plots) do
			if itorchplot.eptensor then
				x = torch.linspace(1,self.lastSequence,self.lastSequence)
				y = itorchplot.epdata[{{1,self.lastSequence,self.lastSequence}}]
				
			else	
			
				 x = itorchplot.xdata:ordered_seq(self.lastSequence)
				if itorchplot.xdim then
					for i = 1,#itorchplot.xdim do
						x = x:index(i+1,itorchplot.xdim[i])
					end
				end
				 y = itorchplot.ydata:ordered_seq(self.lastSequence)
				if itorchplot.ydim then
					for i = 1,#itorchplot.ydim do
						y = y:index(i+1,torch.LongTensor({itorchplot.ydim[i]}))
					end
				end	
			end
			if (x and y and x:size()[1]==y:size()[1] and x:size()[1] > 2) then
				if itorchplot.plotdrawn then 
					itorchplot.plot:line(x:squeeze(), y:squeeze(),'red',''):redraw()		
				else
					itorchplot.plot:line(x:squeeze(), y:squeeze(),'red',''):draw()	
					itorchplot.plotdrawn = true
				end
			end
		end
		-- GNU plots
		if not(SUPRESS_VISUAL_OUTPUT) then 
			for index, plot in ipairs(self.gnuplots) do
				if not(not(plot.dontskipme) or plot.skipcount and self.lastSequence%plot.skipcount > 0) then
					if plot.eptensor then
						-- TODO adapt 
						gnuplot.figure(plot.figureid)
						gnuplot.plot(plot.epdata[{{1,self.lastSequence,self.lastSequence}}])
						--x = torch.linspace(1,self.lastSequence,self.lastSequence)
						--y = itorchplot.epdata[{{1,self.lastSequence,self.lastSequence}}]
					
					else	
						gnuplot.figure(plot.figureid)
						x = (plot.xdata:ordered_seq(self.lastSequence)):squeeze()
						y = (plot.ydata:ordered_seq(self.lastSequence)):squeeze()
						if #(y:size())<=2 then -- 1 or more 1D signals in time
							p = torch.cat(x,y,2)
						elseif #(y:size())>=3 then 
							print("Data too high dimensional, plotting not yet implemented" )
						end
						--gnuplot.plot({'1',x:squeeze(),y[{{},{1}}]:squeeze()},{'2',x:squeeze(),y[{{},{2}}]:squeeze()})
						gnuplot.plot(p)
					end
				end
			end		
		end
		
		
		
		self.lastSequence = sequence_index
	end
end

local plot = torch.class('drl_diagnostics_plot')

-- drl_diagnostics_plot is a parent class, should not ne used directly
function plot:__init( diagnostics, settings )
	self.diagnostics = diagnostics
	assert(settings, "no settings provided")
	assert(settings.name, "no name provided")
	assert(settings.type and type(settings.type)=='string', "no plot type (gnu/itorch) provided")
	assert(settings.update_period_sequence or settings.update_period_timestep, "Provide either update_period_timestep or update_period_sequence")
	self.name = settings.name
	self.update_period_sequence = settings.update_period_sequence
	self.update_period_timestep = settings.update_period_timestep
	self.variables = settings.variables
	self.signals = {}
	self.update_counter = 0
  self.last_seq_update = 0
	if (settings.type == 'gnu') then
		self.gnu = {}
		self.gnu.figureid  = #diagnostics.gnuplots + 1
		if settings.figureid == 1 then
			gnuplot.closeall()
		end
		gnuplot.figure(self.gnu.figureid)	
		gnuplot.title(self.name)
		if(settings.xlabel) then gnuplot.xlabel(settings.xlabel) end
		if(settings.ylabel) then gnuplot.ylabel(settings.ylabel) end
		if(settings.zlabel) then gnuplot.zlabel(settings.zlabel) end
		if(settings.bounds) then
			local xmin = tostring(settings.bounds.xmin or '')
			local ymin = tostring(settings.bounds.ymin or '')
			local xmax = tostring(settings.bounds.xmax or '')
			local ymax = tostring(settings.bounds.ymax or '')
			gnuplot.axis({xmin,xmax,ymin,ymax})
		end
		if settings.logscale then
			gnuplot.raw('set logscale y')
      gnuplot.plotflush()
		end
		table.insert(self.diagnostics.gnuplots,{name = self.name, plot = self})
	elseif settings.type == 'itorch' then
	    assert(false, 'itorch not yet implemented, use teh old functons for now')
	else
		assert(false, 'unknown plot type, supported are: gnu or itorch')
	end	
end

function plot:update( time_index, sequence_index )
	if (not(self.memory_capacity) or self.update_counter < self.memory_capacity) and ((self.update_period_timestep and time_index % self.update_period_timestep  == 0) or (self.update_period_sequence and  sequence_index % self.update_period_sequence  == 0 and not(self.last_seq_update == sequence_index))) then
		
		self.last_seq_update = sequence_index
		if self.update_period_sequence then
      sequence_index = sequence_index -1 -- we want information from the last sequence
    end
    if sequence_index > 0 then
    	self.update_counter = self.update_counter + 1
      for i,signal in ipairs(self.signals) do  
        signal.update( self.update_counter , time_index , sequence_index )
      end
      self:plot()
		end
	end
end

function plot:add_signal( signal )
	assert(signal,'no signal was given')
	-- if the signal is a generalized state
	if (signal.ydata and class.istype(signal.ydata, 'generalized_state')) then
		table.insert(self.diagnostics.monitored_states,signal.ydata)
    signal.yvalues = {}
		if not (signal.getyvalue) then  
      signal.getyvalue = function (self, counter, time_index, sequence_index) 
        signal.yvalues = signal.ydata:ordered_seq(sequence_index) 
      end 
		else
			assert(false, 'not yet implemented')
		end	

		if (signal.xdata and class.istype(signal.xdata, 'generalized_state')) then
			table.insert(self.diagnostics.monitored_states,signal.xdata)
      
      signal.xvalues = {}
			if not (signal.getxvalue) then
				signal.getxvalue = function (self, counter, time_index, sequence_index) 
          signal.xvalues = signal.xdata:ordered_seq(sequence_index) 
        end 
			else
				assert(false, 'not yet implemented')
			end	
		end


	else -- signal is not a generalized state	
		assert(signal.name, 'no signal name was given')
		assert(signal.datapoints, 'number of datapoints not given') 
		--assert(signal.update, 'no function to get the data')
  end
  table.insert(self.signals,signal)
end

function plot:plot()
	assert(false,'This function should have been overloaded')
end


local lp, parent = torch.class('drl_diagnostics_lineplot','drl_diagnostics_plot')

function lp:__init(diagnostics,settings)
	parent.__init(self, diagnostics, settings)
	self.plotsignals = {}
end

function lp:add_signal( signal )
	parent.add_signal(self,signal)
	signal.xvalues = signal.xvalues or torch.Tensor(signal.datapoints)
	signal.yvalues = signal.yvalues or torch.Tensor(signal.datapoints)
  assert(signal.getyvalue, 'the signal should at least have a getyvalue(time_index, sequence_index) function')
	if not(signal.getxvalue) then
		signal.xvalues = torch.linspace(1,signal.datapoints,signal.datapoints)
		signal.getxvalue = function(signal, i, time_index, sequence_index) return signal.xvalues[i] end
	end
  signal.update = function(  counter, time_index, sequence_index )
		if signal.xdata then signal:getxvalue(counter,time_index, sequence_index ) else signal.xvalues[counter] = signal:getxvalue(counter,time_index, sequence_index ) end
		if signal.ydata then signal:getyvalue(counter,time_index, sequence_index ) else signal.yvalues[counter] = signal:getyvalue(counter,time_index, sequence_index ) end
	end
	table.insert(self.plotsignals,{signal.name, signal.xvalues, signal.yvalues})
end

function lp:plot(  )
	if self.gnu then
		self.plotsignals = {}
		for i,signal in ipairs(self.signals) do
      if (signal.xvalues and signal.xvalues:nElement() > 5 and signal.yvalues and signal.yvalues:nElement() > 5) then
        if #signal.yvalues:size() == 1 then
           table.insert(self.plotsignals,{signal.name, signal.xvalues:squeeze(), signal.yvalues:squeeze()})	
        else
          for j =1,signal.yvalues:size(2) do
            table.insert(self.plotsignals,{signal.name .. j, signal.xvalues:squeeze(), signal.yvalues[{{},j}]})	
          end          
        end
			end
		end

		gnuplot.figure(self.gnu.figureid)
		if #self.plotsignals >= 1 then  gnuplot.plot(unpack(self.plotsignals)) end
	else
		assert(false, 'not yet implemented')
	end
end

function diag:addLinePlot(settings)
	local plot = drl_diagnostics_lineplot(self,settings)
	table.insert(self.plots,plot)
	return plot
end

-- intensity map to be used for diagnostics of e.g.: policy map over time, state plase plane visit count, etc. Comes with a handy GNUplot plot function.
-- TODO: make child of plot
local im = torch.class('intensity_map')
---
--[[
settings {
  name [string]: name of the intensity map
  in_dimensionality [number]: number of dimensions of the intensity map (input)
  resolution [LongTensor(d)]: resolutions of the dimensions
  bounds  [Tensor(d,2)]: min and max values per dimension
  outputs [optional number]: number of intensity maps (outputs)
  memory_capacity  [number]: number of intensity map timesteps
  update_period_timestep [optional number]: Update every n timesteps
  update_period_sequence [optional number]: Update every n sequences
  update_function [function(time_index, sequence_index)]: function to be called when updating (don't call directly, use intensity_map:update(time_index, sequence_index) instead).
  variables [optional table]: table that will be accessible through intensity_map.variables
  plot_options [optional table, supply to get plots]
    type [string: <mesh> or <image>]: plot type
    
  
--]]

function calculate_TDE(experience_memory,settings)
  TDET = experience_memory:get_extra_info("TDE")  
  updateIndex = experience_memory:get_extra_info("UPDATE")
  if settings.since=="last_update" then
    local last_update = torch.max(updateIndex)
    return TDET[updateIndex:eq(last_update)]:mean()
  end
end

function calculate_AVGQ(experience_memory,settings)
  QP = experience_memory:get_extra_info("QPRED")
  QFP = experience_memory:get_extra_info("QFPRED")
  updateIndex = experience_memory:get_extra_info("UPDATE")
  if settings.since=="last_update" then
    local last_update = torch.max(updateIndex)
    return {QP[updateIndex:eq(last_update)]:mean() , QFP[updateIndex:eq(last_update)]:mean()}
  end
end




function im:__init(diag,settings)
  local check_input = function(input)  
    assert(settings,"Settings are nil")
    assert(settings.name and type(settings.name=="string"),"Provide a name for the intensity map")
    assert(settings.in_dimensionality,"Settings.in_dimensionality = nil, provide the dimensionality of the intensity map")
    assert(type(settings.in_dimensionality)=='number',"Settings.in_dimensionality should be a number (the number of dimensions)")
    assert(settings.resolution,"Provide the resolutions of the dimensions of the intensity map: settings.resolution")
    assert(torch.type(settings.resolution)=="torch.LongStorage","Provide the resolutions of the dimensions as a LongStorage")
    assert(settings.resolution:size()==settings.in_dimensionality,"settings.resolution should be a 1D tensor with n values where n is the dimensionality of the intensity map")
    assert(settings.bounds and (#settings.bounds:size())==2 and settings.bounds:size(2)==settings.in_dimensionality and settings.bounds:size(1)==2,"settings.bounds incorrect, should be a tensor of size 2xd" )
    assert(settings.memory_capacity and type(settings.memory_capacity)=="number","settings.memory_capacity not specified (as a number)")
  end
  
  check_input(settings)
  self.name             = settings.name
  self.dimensionality   = settings.in_dimensionality
  self.resolution       = settings.resolution
  self.bounds           = settings.bounds
  self.outputs          = settings.outputs or 1
  self.memory_capacity  = settings.memory_capacity 
  self.update_function  = settings.update_function
  self.plot_options     = settings.plot_options
  self.last_seq_update  = 0
  if (settings.update_period_timestep) then
    self.update_period_timestep = settings.update_period_timestep
  else
    if settings.update_period_sequence then
      self.update_period_sequence = settings.update_period_sequence
    else
      self.update_period_sequence = 1
      print(self.name .. " did not provide an update period, updating once per sequence")
    end
  end
  self.variables = settings.variables
  self.update_counter = 0
  self.memsize = torch.LongStorage(2+#self.resolution)
  self.memsize[1] = self.memory_capacity
  self.memsize[2] = self.outputs
  for i = 3,2+self.dimensionality do
    self.memsize[i] = self.resolution[i-2]
  end
  self.memory = torch.Tensor(self.memsize)
  self.values = {}
  for dim = 1,self.dimensionality do
    table.insert(self.values, torch.linspace(self.bounds[1][dim],self.bounds[2][dim],self.resolution[dim]))
  end
  
  -- plot 
  if settings.plot_options then
    self.plot_options = settings.plot_options
    self.plot_options.figures = {}
    for p = 1,self.outputs do 
      figureid  = #diag.gnuplots + 1
      table.insert(diag.gnuplots,{})
      table.insert(self.plot_options.figures,figureid)
      gnuplot.figure(figureid)
      gnuplot.title(self.name .. " - " .. p )
      if (settings.state_names) then
        gnuplot.xlabel(self.settings.state_names[1])
        gnuplot.ylabel(self.settings.state_names[2])
    	end
    end
  end
  
  table.insert(diag.intensity_maps,self)
end

function im:update(time_index,sequence_index)
  if self.update_counter < self.memory_capacity and ((self.update_period_timestep and time_index % self.update_period_timestep  == 0) or (self.update_period_sequence and  sequence_index % self.update_period_sequence  == 0 and not(self.last_seq_update == sequence_index))) then
    self.last_seq_update = sequence_index
    self.update_counter = self.update_counter + 1
    self:update_function(time_index,sequence_index,self.update_counter)
    for out=1,self.outputs do
      self:plot(self.update_counter,out)
    end
    if (self.update_counter == self.memory_capacity) then
      print(self.name .. " has reached its memory capacity and will no longer update")
    end
  end  
end


function im:plot(time_index, out_index)
  if (self.dimensionality ==2) then
      gnuplot.figure(self.plot_options.figures[out_index])
      if self.plot_options.type=="image" then
        gnuplot.imagesc(self.memory[time_index][out_index],'color')
      elseif self.plot_options.type=="mesh" then
        if self.plot_options.ground_truth_values and self.plot_options.ground_truth_points then
          gnuplot.splot({self.plot_options.ground_truth_points[1],self.plot_options.ground_truth_points[2],self.plot_options.ground_truth_values[out_index]},{self.memory[time_index][out_index]})
        else
          gnuplot.splot(self.memory[time_index][out_index])
        end
      end     
  else
    print("Plotting " .. self.dimensionality .."D data not implemented")
  end
end


local function deepTableCopy(original)
    local copy = {}
    for k, v in pairs(original) do
        if type(v) == 'table' then
            v = deepTableCopy(v)
        end
        if torch.isTensor(v) then
        	v = v:clone()
        end
        copy[k] = v
    end
    return copy
end

-- This class prevents doing double work for intensity maps based on the location of database samples in the state(-action) space.
local distmap = torch.class('distribution_map')

function distmap:__init( diag, experience_memory, settings  )	
	assert(experience_memory,'experience_memory not given')
	assert(diag, "diagnostics not given")
	assert(settings.maps,"provide one or more of the following strings in a table called maps: 'STATE',TDE'")
	-- namen voor im mappen op basis van maps
	self.lastUpdate = 0
	self.experience_memory = experience_memory
	self.ims = {}
	for i,name in ipairs(settings.maps) do
		mapsettings = deepTableCopy(settings)
		mapsettings.name = name
		mapsettings.update_function = function ( im, time_index,sequence_index, update_counter )
			self:update(time_index,sequence_index, update_counter, im)
		end
		self.ims[name] = intensity_map(diag, mapsettings)
	end
	self.settings = settings
	self.bounds = settings.bounds
	self.occ_out_of_range = 0
	self.occurancemap = torch.Tensor(settings.resolution)	
  self.cells = self.occurancemap:nElement()
  self.scale = {}
  self.scale.sub = torch.Tensor(1,settings.bounds:size(2))
  self.scale.mul = torch.Tensor(1,settings.bounds:size(2))
  for dimension = 1,settings.bounds:size(2) do 
  	self.scale.sub[1][dimension] = settings.bounds[1][dimension]
  	self.scale.mul[1][dimension] = settings.resolution[dimension] / (settings.bounds[2][dimension] - settings.bounds[1][dimension])
  end

end

function distmap:update( time_index,sequence_index, update_counter, intensity_map )
	if update_counter > self.lastUpdate then
		self.lastUpdate = update_counter
		-- updateIndex = experience_memory:get_extra_info("UPDATE")
		-- TDET = experience_memory:get_extra_info("TDE")	
		states = self.experience_memory.experience_database.state[1][{{1,math.max(self.experience_memory.experience_database.last_write_index,10)}}]
		-- TODO: Remove this expensive check!
		statesBefore = states:clone()
		
		self.occurancemap:zero()
		self.occ_out_of_range = 0

		for name,instymap in pairs(self.ims) do
				if name == 'TDE' then
						instymap.currentSpecialValues  = self.experience_memory:get_extra_info("TDE")	            
				elseif name == 'STATE' then
						
				elseif name == 'USECOUNT' then 

				else  
					assert(false, 'Unknown map: '  .. name)
				end
				
		end

		local indices = self:stateindexes(states)
		local currentIndex = torch.LongStorage(states:size(2))
		local longCurrentIndex = torch.LongStorage(states:size(2)+2)

		longCurrentIndex[1] = update_counter
		longCurrentIndex[2] = 1
		
		for dbentry = 1,states:size(1) do
			local outofbouds = false
			local calcindx = 0

      for dim = 1,indices:size(2) do
				calcindx = indices[dbentry][dim]
				if calcindx < 1 or calcindx > self.settings.resolution[dim] then
					outofbouds = true
				end
				currentIndex[dim] = calcindx
				longCurrentIndex[dim+2]  = calcindx 
			end
			if outofbouds then
				self.occ_out_of_range = self.occ_out_of_range + 1
			else
				self.occurancemap[currentIndex] = self.occurancemap[currentIndex] + 1
				for i,instymap in pairs(self.ims) do
					if instymap.currentSpecialValues then
						instymap.memory[longCurrentIndex] = instymap.memory[longCurrentIndex] + instymap.currentSpecialValues[dbentry]
					end
				end
			end
		end

		local normalizemap = self.occurancemap:clone()
		normalizemap:clamp(1,math.huge)
		outprc = 100*self.occ_out_of_range/states:size(1)
		if outprc > 1 then
			print('Distribution map: ' .. outprc .. '% of the samples are outside the bounds of the map.') 
		end
		for name,instymap in pairs(self.ims) do
				if name == 'STATE' then
						instymap.memory[update_counter][1]:copy( self.occurancemap * (self.cells / states:size(1)))
						print(self.occurancemap:sum() .. ' ' .. states:size(1))            
				else
						instymap.memory[update_counter][1]:copy( (torch.cdiv(instymap.memory[update_counter][1],normalizemap) ))
				end
		end
		-- TODO: Remove this expensive check!
		assert((statesBefore - states):abs():max()==0,'states changed!')
	end

end

function distmap:stateindexes( states, actions )
	assert(not(actions),'action based indexing not yet implemented')
	local minsub = torch.expand(self.scale.sub, states:size(1), states:size(2))
	local mul = torch.expand(self.scale.mul, states:size(1), states:size(2))
	local indices = (torch.ones(states:size()):addcmul(states - minsub,mul)):floor()
	return indices
end
