--[[

		Tim de Bruin 2016
		deepRL-torch
		Experiments based on the matlab versions by Robert Babuska
--]]



local experiment = torch.class('drl_experiment')

local ode4_ti = function(odefun,tspan,y0)
	--h = diff(tspan)
	local hi = tspan[2] - tspan[1] -- asume constant time interval
	local neq = y0:nElement()
	local N = #tspan
	local Y = torch.Tensor(N,neq):zero()
	local F = torch.Tensor(4,neq):zero()
	Y[1]:copy(y0)
	for i = 2,N do
	  local yi = Y[i-1]:clone()
	  F[1]:copy(odefun(yi))
	  F[2]:copy(odefun(yi:clone():add(F[1]:clone():mul(0.5*hi))))
	  F[3]:copy(odefun(yi:clone():add(F[2]:clone():mul(0.5*hi))))
	  F[4]:copy(odefun(yi:clone():add(F[3]:clone():mul(hi)))) 
	  Y[i]:copy( yi:clone():add( (F[1]:clone():add(F[2]:clone():mul(2)):add(F[3]:clone():mul(2)):add(F[4]:clone()) ):mul(hi/6)) )
	end
	return Y
end

function experiment:__init( settings )

	self.environments = {
		magman = {
			params = {Ts = 0.02, ref = torch.Tensor({0.035,0})},
			f = function (env, state, action)
				local y = ode4_ti(env.eom, {0, env.params.Ts/2, env.params.Ts}, torch.cat(state, action))
				local next_state = y[{{y:size(1)},{1,2}}]
				return next_state
			end,
			eom = function ( x )
				-- Equations of motion for the DCSC magnetic manipulation setup
				local alpha = 5.52e-10--   % magnetic force function parameter
				local beta = 1.75e-4--     % magnetic force function parameter
				local b = 0.0161--         % viscous friction coefficient
				local m = 0.032--          % ball mass
				                                            
				local dx = torch.Tensor(x:size()):zero()
				dx[1] = x[2]
				local term4 = 0
				if x:nElement() == 6 then
					term4 = x[6]*(-alpha*(x[1]-0.075)/((x[1]-0.075)^2 + beta)^3)
				end
				dx[2] = -(b/m)*x[2] + 
							 (x[3]*(-alpha*(x[1]-0.000)/((x[1]-0.000)^2 + beta)^3) + 
			          x[4]*(-alpha*(x[1]-0.025)/((x[1]-0.025)^2 + beta)^3) + 
			          x[5]*(-alpha*(x[1]-0.050)/((x[1]-0.050)^2 + beta)^3) + term4 )/m
				return dx 
			end,
			r = function (env, state,action)
				local	Qdiag = {10,0.5}     --% diagonal of the reward Q matrix
				local Pdiag = {0, 0, 0, 0  };   --! drda!    --% diagonal of the reward P matrix
				local reward = 0
				for i = 1,state:nElement() do
					reward = reward - math.abs(state[i] - env.params.ref[i])*Qdiag[i]
				end
				for i = 1,action:nElement() do
					reward = reward - math.abs(action[i])*Pdiag[i]
				end
				return reward
			end,
			drda = function (env, state, action) -- for four magnets!
				return action:clone():zero()
			end,
			startstate = torch.Tensor({0,0}),
			alternativestartstate = {	torch.Tensor({0.07,0}),
																torch.Tensor({0.035,0.1}),
																torch.Tensor({0.035,-0.1}),
																torch.Tensor({0,0.3}),
															},
			magwalled = true,
			minu = {0,0,0},
			maxu = {0.6,0.6,0.6},
			mins = torch.Tensor({-0.035, -0.4}),
			maxs = torch.Tensor({0.105, 0.4}),
		},
		pendulum = {
			params = {Ts = 0.02, ref = torch.Tensor({math.pi,0}), SEQLENGTH = 200},
			f = function (env, state, action)
				local y = ode4_ti(env.eom, {0, env.params.Ts/2, env.params.Ts}, torch.cat(state, action))
				local next_state = y[{{y:size(1)},{1,2}}]
				if env.wrapflag then
          if next_state[1][1] > math.pi then next_state[1][1] = next_state[1][1] - 2*math.pi end
          if next_state[1][1] < -math.pi then next_state[1][1] = next_state[1][1] + 2*math.pi end
        end
				return next_state
			end,
			eom = function ( x )
				-- Equations of motion for DCSC inverted pendulum setup
				local J = 1.91e-4 --    % Pendulum inertia
				local M = 5.5e-2 --     % Pendulum mass
				local g = 9.81--       % Gravity constant
				local l = 4.2e-2--     % Pendulum length
				local b = 3e-6--       % Viscous damping
				local K = 5.36e-2--    % Torque constant
				local R = 9.5--        % Rotor resistance
				                                            
				local dx = torch.Tensor(x:size()):zero()
				dx[1] = x[2]
				dx[2] = (-M*g*l*math.sin(x[1]) - (b + K^2/R)*x[2] + K/R*x[3])/J
				dx[3] = 0
				return dx 
			end,
			r = function (env, state,action)
				local	Qdiag = {5,0.1}     --% diagonal of the reward Q matrix
				local Pdiag = {1};      --% diagonal of the reward P matrix
				local reward = 0
				if env.wrapflag then
					state = state:clone():abs()
				end
				for i = 1,state:nElement() do
					reward = reward - (math.abs(state[i] - env.params.ref[i]))^1*Qdiag[i]
				end
				for i = 1,action:nElement() do
					reward = reward - (math.abs(action[i]))^1*Pdiag[i] --! drda!
				end
				return reward/100
			end,
			drda = function (env, state, action)
				return action:clone():fill(-1)-- linear with P = 1
			end,
			wrapflag = true,
			startstate = torch.Tensor({0.0,0}),
			alternativestartstate = {	torch.Tensor({math.pi/2,20}),
																torch.Tensor({math.pi/2,-20}),
																torch.Tensor({-math.pi/2,20}),
																torch.Tensor({-math.pi/2,-20}),
															},
			minu = {-3},
			maxu = {3},
			mins = torch.Tensor({-math.pi, -30}),
			maxs = torch.Tensor({math.pi, 30}),
		},
	}

	if settings.environment == 'magman' then
		self.environment = self.environments.magman
		self.environment.state_dimension = 2
		if settings.discrete then 
			self.discrete_actions = torch.Tensor(64,3)
			for a = 0,63 do
				self.discrete_actions[a+1][1] = (a%4)*0.2
				self.discrete_actions[a+1][2] = (math.floor(a/4)%4)*0.2
				self.discrete_actions[a+1][3] = (math.floor(a/16)%4)*0.2
			end
			self.environment.action_type = 'DISCRETE'
			self.environment.action_dimension = 64
		else
			self.environment.action_type = 'CONTINUOUS'
			self.environment.action_dimension = settings.magnets
			if settings.magnets == 4 then
				self.environment.minu[4] = self.environment.minu[3]
				self.environment.maxu[4] = self.environment.maxu[3]
			end
		end
		self.action = torch.Tensor(settings.magnets):zero()
	elseif settings.environment == 'pendulum' then
		self.environment = self.environments.pendulum
		if settings.bwddiff then
			self.environment.f= function (env, state, action)
				local y = ode4_ti(env.eom, {0, env.params.Ts/2, env.params.Ts}, torch.cat(state, action))
				local next_state = y[{{y:size(1)},{1,2}}]
				next_state[1][2] = (next_state[1][1]-state[1])/(env.params.Ts)
				if env.wrapflag then
          if next_state[1][1] > math.pi then next_state[1][1] = next_state[1][1] - 2*math.pi end
          if next_state[1][1] < -math.pi then next_state[1][1] = next_state[1][1] + 2*math.pi end
        end
				return next_state
			end
		end
		if settings.altrew then 
			print("USING ALTERNATIVE REWARD FUNCTION!")
			self.environment.r = function (env, state,action)
				local	Qdiag = {5,0.1} --5 0    --% diagonal of the reward Q matrix
				
				local Pdiag = {0}; --%1     --% diagonal of the reward P matrix
				local reward = 0
				if env.wrapflag then
					state = state:clone():abs()
				end
				for i = 1,state:nElement() do
					reward = reward - (math.abs(state[i] - env.params.ref[i]))^1*Qdiag[i]
				end
				for i = 1,action:nElement() do
					reward = reward - (math.abs(action[i]))^1*Pdiag[i] --! drda!
				end
				return reward/100
			end
		end 
		self.environment.state_dimension = 2
		if settings.discrete then 
			self.environment.action_dimension = 16
			self.discrete_actions = torch.linspace(self.environment.minu,self.environment.maxu,self.environment.action_dimension)
			self.environment.action_type = 'DISCRETE'		
		else
			self.environment.action_type = 'CONTINUOUS'
			self.environment.action_dimension = 1
		end
		self.action = torch.Tensor(1):zero()
	end

	self.state 	= self.environment.startstate:clone()
	self.reward = 0
	self.steps  = 0
	self.terminal = 0

	self.environment.params.Ts =  settings.ts or self.environment.params.Ts
	print("Started the " .. settings.environment .. " environment with a sample time of " .. self.environment.params.Ts .. ' seconds.')
end

function experiment:check_action(action)
	if self.environment.action_type == 'DISCRETE' and not(type(action) == number) then
		if action:nElement()==1 then
			action = action[1]
		else
			local av,ai = torch.max(action,1)
			action = ai
		end
	end

	if type(action) == 'number' and self.environment.action_type == 'DISCRETE' then 
		return self.discrete_actions[action]:clone()
	else
		return action
	end
end



function experiment:step( action )
	self.action:copy(self:check_action(action))
	self.state:copy(self.environment:f( self.state, self.action ))
	self.reward = self.environment:r(self.state, self.action)
	if self.environment.magwalled then 
		if self.state[1] < -0.035 then
			self.state[1] = -0.035
			self.state[2] = 0.001
			self.reward = self.reward - 1
		end
		if self.state[1] > 0.105 then
			self.state[1] = 0.105
			self.state[2] = -0.001
			self.reward = self.reward - 1
		end
	end
	self.steps = self.steps + 1
	--[[
	local maxsteps = self.environment.params.SEQLENGTH or 100
	if self.steps >= maxsteps then 
		self.terminal = 3
	end
	]]
end

function experiment:reset(scenario)
	if scenario then
		self.state:copy(self.environment.alternativestartstate[scenario])
	else
		self.state:copy(self.environment.startstate)
	end
  --self.state:copy(self:uniform_random_state())
	
	self.reward 	= 0
	self.steps 		= 0
	self.terminal = 0	
end

function experiment:get_state()
	
	return self.state:clone()

end

function experiment:set_state( state, normalized )
	if normalized then
		if #state:size() == 1 then
			state = state:resize(1,2)
		end
		state = self:states_denormalize(state)
	end
	self.state = state:reshape(2)
end

function experiment:get_reward()
	return self.reward
end

function experiment:get_terminal()
	return self.terminal
end

function experiment:get_bounds()
	return {
		state = {
			min = self.environment.mins,
			max = self.environment.maxs,
			dimension = self.environment.state_dimension,
		},
		action = {
			min = self.environment.minu,
			max = self.environment.maxu,
			dimension = self.environment.action_dimension,
			action_type = self.environment.action_type,
		},		
	}
end

function experiment:reward_function(state, action, next_state)
	local corrected_action = self:check_action(action)
	return self.environment:r(next_state,corrected_action)
end

function experiment:transition_function(state, action)
	local corrected_action = self:check_action(action)
	local next_state = self.environment:f( state, action ) 
	local reward_correction = 0
	if self.environment.magwalled then 
		next_state = torch.Tensor(2):copy(next_state)
		if next_state[1] < -0.035 then
			next_state[1] = -0.035
			next_state[2] = 0.001
			reward_correction =  - 1
		end
		if next_state[1] > 0.105 then
			next_state[1] = 0.105
			next_state[2] = -0.001
			reward_correction = - 1
		end
	end
	return next_state, reward_correction
end

function experiment:states_denormalize(states)
	local nstates = states:clone()
	for i = 1,states:size(2) do
		nstates[{{},i}]:mul(0.5*(self.environment.maxs[i]-self.environment.mins[i]))
		nstates[{{},i}]:add(0.5*(self.environment.maxs[i]+self.environment.mins[i]))
	end
	return nstates
end

function experiment:states_normalize(states)
	local nstates = states:clone()
	for i = 1,states:size(2) do
		nstates[{{},i}]:add(-0.5*(self.environment.maxs[i]+self.environment.mins[i]))
		nstates[{{},i}]:div(0.5*(self.environment.maxs[i]-self.environment.mins[i]))
	end
	return nstates
end

function experiment:actions_denormalize(actions)
	local nactions = actions:clone()
	for i = 1,actions:size(2) do
		nactions[{{},i}]:mul(0.5*(self.environment.maxu[i]-self.environment.minu[i]))
		nactions[{{},i}]:add(0.5*(self.environment.maxu[i]+self.environment.minu[i]))
	end
	return nactions
end

function experiment:actions_normalize(actions)
	local nactions = actions:clone()
	for i = 1,actions:size(2) do
		nactions[{{},i}]:add(-0.5*(self.environment.maxu[i]+self.environment.minu[i]))
		nactions[{{},i}]:div(0.5*(self.environment.maxu[i]-self.environment.minu[i]))
	end
	return nactions
end

function experiment:forward(statesandactions)
	local states = statesandactions[1]  
	local actions = statesandactions[2]  
	states = self:states_denormalize(states)
	actions = self:actions_denormalize(actions)
	local next_states = states:clone()
	for i=1,states:size(1) do
		local tempstate = states[i]:clone()
		local tempaction = self:check_action(actions[i]:clone())
		next_states[i]:copy(self.environment:f( tempstate, tempaction ))
		if self.environment.magwalled then 
			if next_states[i][1] < -0.035 then
				next_states[i][1] = -0.035
				next_states[i][2] = 0.001
			end
			if next_states[i][1] > 0.105 then
				next_states[i][1] = 0.105
				next_states[i][2] = -0.001
			end
		end
	end
	next_states = self:states_denormalize(next_states)
	return next_states
end

-- finite diff, only calculated for the actions!
function experiment:backward(statesandactions, multiplier)
	local states = statesandactions[1]:clone()  
	local actions = statesandactions[2]:clone()  
	local DELTA = 0.001
	local sp0 = self:forward(statesandactions)
	local aderiv = actions:clone()
	for i = 1,actions:size(2) do
		local adelta = actions:clone()
		adelta[{{},i}]:add(DELTA)
		local spplus = self:forward({states,adelta})
		local spdelta = spplus:clone():add(-1,sp0)
		spdelta:div(DELTA):cmul(multiplier)
		aderiv[{{},i}]:copy(spdelta:sum(2))
	end
	return {sp0:zero(), aderiv}
end


function experiment:uniform_random_state()
	local s = self.state:clone()
	for i=1,s:nElement() do
		s[i] = self.environment.mins[i] + math.random() * (self.environment.maxs[i] - self.environment.mins[i])
	end
	return s
end

function experiment:drda(state, action)
	return self.environment:drda(state,action)
end